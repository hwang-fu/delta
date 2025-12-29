use delta::tensor::Tensor;

fn main() {
    println!("=== 1D Vector ===");
    let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    println!("{}", v);

    println!("\n=== 2D Matrix ===");
    let m = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("{}", m);

    println!("\n=== 3D Tensor ===");
    let t = Tensor::zeros(&[2, 3, 4]);
    println!("{}", t);

    println!("\n=== Large Matrix (truncation) ===");
    let big = Tensor::zeros(&[10, 10]);
    println!("{}", big);
}
