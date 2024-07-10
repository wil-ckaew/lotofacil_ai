use std::error::Error;
use csv::ReaderBuilder;
use tch::{nn, Device, Tensor, Kind, Reduction};
use tch::nn::Module;
use tch::nn::OptimizerConfig;

fn main() -> Result<(), Box<dyn Error>> {
    // Carregar dados do CSV
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("resultado.csv")?;

    // Ler dados do CSV para um vetor de vetores de f32
    let mut data: Vec<Vec<f32>> = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let row: Vec<f32> = record.iter().map(|s| s.parse::<f32>().unwrap()).collect();
        data.push(row);
    }

    // Converter para um tensor
    let data_tensor = Tensor::of_slice(&data.iter().flatten().copied().collect::<Vec<f32>>().as_slice())
        .view([-1, 15]);

    // Exemplo de modelo simples em Rust, similar ao modelo em Python
    let vs = nn::VarStore::new(Device::Cpu);
    let model = simple_nn(&vs.root());

    // Definir targets com base nos dados do próximo concurso
    let targets = data_tensor
        .narrow(0, 0, 1)
        .view([-1])
        .to_kind(Kind::Float);

    // Definir critério de perda e otimizador
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3)?;

    // Treinamento com os dados do CSV
    for epoch in 1..=1000 {
        optimizer.zero_grad();
        let outputs = model.forward(&data_tensor);
        let loss = outputs.mse_loss(&targets, Reduction::Mean);
        loss.backward();
        optimizer.step();

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.double_value(&[]));
        }
    }

    // Salvar o modelo treinado
    vs.save("model.pth")?;
    println!("Modelo treinado e salvo com sucesso!");

    // Gerar previsões usando o modelo treinado
    let predictions = model.forward(&data_tensor);

    // Extrair a última previsão (última linha do tensor de previsões)
    let last_prediction = predictions.get(predictions.size()[0] - 1);

    // Converter a previsão para um vetor de f32 e exibir
    let result: Vec<f32> = last_prediction.into();
    println!("Previsão dos próximos 15 números: {:?}", result);

    Ok(())
}

// Definir modelo simples em Rust, similar ao modelo em Python
fn simple_nn(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "fc1", 15, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "fc2", 128, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "fc3", 128, 15, Default::default()))
}
