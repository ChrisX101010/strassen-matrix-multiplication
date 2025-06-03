use std::time::Instant;
use std::sync::Arc;
use std::thread;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }
    
    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Matrix { data, rows, cols }
    }
    
    pub fn random(rows: usize, cols: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data = (0..rows)
            .map(|_| (0..cols).map(|_| rng.gen_range(-100.0..100.0)).collect())
            .collect();
        Matrix { data, rows, cols }
    }
    
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }
    
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i][j] = value;
    }
    
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
        result
    }
    
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) - other.get(i, j));
            }
        }
        result
    }
    
    // Extract submatrix
    pub fn submatrix(&self, start_row: usize, end_row: usize, start_col: usize, end_col: usize) -> Matrix {
        let rows = end_row - start_row;
        let cols = end_col - start_col;
        let mut result = Matrix::new(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                result.set(i, j, self.get(start_row + i, start_col + j));
            }
        }
        result
    }
    
    // Copy submatrix into this matrix at specified position
    pub fn copy_submatrix(&mut self, submatrix: &Matrix, start_row: usize, start_col: usize) {
        for i in 0..submatrix.rows {
            for j in 0..submatrix.cols {
                self.set(start_row + i, start_col + j, submatrix.get(i, j));
            }
        }
    }
    
    // Pad matrix to nearest power of 2
    pub fn pad_to_power_of_2(&self) -> Matrix {
        let max_dim = self.rows.max(self.cols);
        let padded_size = max_dim.next_power_of_two();
        
        let mut padded = Matrix::new(padded_size, padded_size);
        for i in 0..self.rows {
            for j in 0..self.cols {
                padded.set(i, j, self.get(i, j));
            }
        }
        padded
    }
    
    // Extract result from padded matrix
    pub fn extract_result(&self, target_rows: usize, target_cols: usize) -> Matrix {
        self.submatrix(0, target_rows, 0, target_cols)
    }
}

pub struct StrassenMultiplier {
    threshold: usize,
    use_parallel: bool,
    num_threads: usize,
}

impl StrassenMultiplier {
    pub fn new() -> Self {
        StrassenMultiplier {
            threshold: 64,  // Switch to standard multiplication below this size
            use_parallel: true,
            num_threads: num_cpus::get(),
        }
    }
    
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }
    
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.use_parallel = parallel;
        self
    }
    
    // Standard matrix multiplication for small matrices or base case
    fn standard_multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.rows);
        let mut result = Matrix::new(a.rows, b.cols);
        
        if self.use_parallel && a.rows > 32 {
            result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..b.cols {
                    let mut sum = 0.0;
                    for k in 0..a.cols {
                        sum += a.get(i, k) * b.get(k, j);
                    }
                    row[j] = sum;
                }
            });
        } else {
            for i in 0..a.rows {
                for j in 0..b.cols {
                    let mut sum = 0.0;
                    for k in 0..a.cols {
                        sum += a.get(i, k) * b.get(k, j);
                    }
                    result.set(i, j, sum);
                }
            }
        }
        result
    }
    
    // Main Strassen multiplication algorithm
    pub fn multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.rows);
        
        // For non-square matrices or small matrices, use standard multiplication
        if a.rows != a.cols || b.rows != b.cols || a.rows < self.threshold {
            return self.standard_multiply(a, b);
        }
        
        let n = a.rows;
        
        // Base case
        if n <= self.threshold {
            return self.standard_multiply(a, b);
        }
        
        // Pad matrices to power of 2 if needed
        let padded_a = if n.is_power_of_two() { a.clone() } else { a.pad_to_power_of_2() };
        let padded_b = if n.is_power_of_two() { b.clone() } else { b.pad_to_power_of_2() };
        
        let result_padded = self.strassen_recursive(&padded_a, &padded_b);
        
        // Extract the actual result
        result_padded.extract_result(a.rows, b.cols)
    }
    
    fn strassen_recursive(&self, a: &Matrix, b: &Matrix) -> Matrix {
        let n = a.rows;
        
        if n <= self.threshold {
            return self.standard_multiply(a, b);
        }
        
        let mid = n / 2;
        
        // Divide matrices into quadrants
        let a11 = a.submatrix(0, mid, 0, mid);
        let a12 = a.submatrix(0, mid, mid, n);
        let a21 = a.submatrix(mid, n, 0, mid);
        let a22 = a.submatrix(mid, n, mid, n);
        
        let b11 = b.submatrix(0, mid, 0, mid);
        let b12 = b.submatrix(0, mid, mid, n);
        let b21 = b.submatrix(mid, n, 0, mid);
        let b22 = b.submatrix(mid, n, mid, n);
        
        // Calculate the 7 Strassen products
        if self.use_parallel && n > 128 {
            // Parallel computation of Strassen products
            let products: Vec<Matrix> = vec![
                (&a11.add(&a22), &b11.add(&b22)),  // M1
                (&a21.add(&a22), &b11),            // M2
                (&a11, &b12.subtract(&b22)),       // M3
                (&a22, &b21.subtract(&b11)),       // M4
                (&a11.add(&a12), &b22),            // M5
                (&a21.subtract(&a11), &b11.add(&b12)), // M6
                (&a12.subtract(&a22), &b21.add(&b22)), // M7
            ].into_par_iter()
            .map(|(x, y)| self.strassen_recursive(x, y))
            .collect();
            
            let m1 = &products[0];
            let m2 = &products[1];
            let m3 = &products[2];
            let m4 = &products[3];
            let m5 = &products[4];
            let m6 = &products[5];
            let m7 = &products[6];
            
            // Calculate result quadrants
            let c11 = m1.add(&m4).subtract(&m5).add(&m7);
            let c12 = m3.add(&m5);
            let c21 = m2.add(&m4);
            let c22 = m1.subtract(&m2).add(&m3).add(&m6);
            
            // Combine quadrants
            let mut result = Matrix::new(n, n);
            result.copy_submatrix(&c11, 0, 0);
            result.copy_submatrix(&c12, 0, mid);
            result.copy_submatrix(&c21, mid, 0);
            result.copy_submatrix(&c22, mid, mid);
            
            result
        } else {
            // Sequential computation
            let m1 = self.strassen_recursive(&a11.add(&a22), &b11.add(&b22));
            let m2 = self.strassen_recursive(&a21.add(&a22), &b11);
            let m3 = self.strassen_recursive(&a11, &b12.subtract(&b22));
            let m4 = self.strassen_recursive(&a22, &b21.subtract(&b11));
            let m5 = self.strassen_recursive(&a11.add(&a12), &b22);
            let m6 = self.strassen_recursive(&a21.subtract(&a11), &b11.add(&b12));
            let m7 = self.strassen_recursive(&a12.subtract(&a22), &b21.add(&b22));
            
            // Calculate result quadrants
            let c11 = m1.add(&m4).subtract(&m5).add(&m7);
            let c12 = m3.add(&m5);
            let c21 = m2.add(&m4);
            let c22 = m1.subtract(&m2).add(&m3).add(&m6);
            
            // Combine quadrants
            let mut result = Matrix::new(n, n);
            result.copy_submatrix(&c11, 0, 0);
            result.copy_submatrix(&c12, 0, mid);
            result.copy_submatrix(&c21, mid, 0);
            result.copy_submatrix(&c22, mid, mid);
            
            result
        }
    }
}

// Benchmark and comparison utilities
pub struct MatrixBenchmark;

impl MatrixBenchmark {
    pub fn compare_algorithms(size: usize, iterations: usize) {
        println!("Matrix Multiplication Benchmark - Size: {}x{}", size, size);
        println!("Iterations: {}", iterations);
        println!("{:-<60}", "");
        
        let multiplier = StrassenMultiplier::new().with_threshold(64);
        
        let mut total_strassen_time = 0.0;
        let mut total_standard_time = 0.0;
        
        for i in 0..iterations {
            println!("Iteration {}/{}", i + 1, iterations);
            
            let a = Matrix::random(size, size);
            let b = Matrix::random(size, size);
            
            // Benchmark Strassen
            let start = Instant::now();
            let _result_strassen = multiplier.multiply(&a, &b);
            let strassen_time = start.elapsed().as_secs_f64();
            total_strassen_time += strassen_time;
            
            // Benchmark Standard
            let start = Instant::now();
            let _result_standard = multiplier.standard_multiply(&a, &b);
            let standard_time = start.elapsed().as_secs_f64();
            total_standard_time += standard_time;
            
            println!("  Strassen: {:.4}s, Standard: {:.4}s", strassen_time, standard_time);
        }
        
        let avg_strassen = total_strassen_time / iterations as f64;
        let avg_standard = total_standard_time / iterations as f64;
        let speedup = avg_standard / avg_strassen;
        
        println!("{:-<60}", "");
        println!("Results:");
        println!("Average Strassen time: {:.4}s", avg_strassen);
        println!("Average Standard time: {:.4}s", avg_standard);
        println!("Speedup: {:.2}x", speedup);
        
        if speedup > 1.0 {
            println!("Strassen is {:.2}x faster!", speedup);
        } else {
            println!("Standard is {:.2}x faster!", 1.0 / speedup);
        }
    }
    
    pub fn memory_usage_analysis(size: usize) {
        println!("Memory Usage Analysis for {}x{} matrices", size, size);
        println!("{:-<50}", "");
        
        let matrix_memory = size * size * 8; // 8 bytes per f64
        let strassen_overhead = matrix_memory * 7; // 7 intermediate matrices
        
        println!("Single matrix memory: {} bytes ({:.2} MB)", 
                matrix_memory, matrix_memory as f64 / 1_048_576.0);
        println!("Strassen overhead: {} bytes ({:.2} MB)", 
                strassen_overhead, strassen_overhead as f64 / 1_048_576.0);
        println!("Total peak memory: {} bytes ({:.2} MB)", 
                matrix_memory * 2 + strassen_overhead, 
                (matrix_memory * 2 + strassen_overhead) as f64 / 1_048_576.0);
    }
}

// Configuration and optimization utilities
pub struct StrassenOptimizer;

impl StrassenOptimizer {
    pub fn find_optimal_threshold(max_size: usize) -> usize {
        println!("Finding optimal threshold for Strassen algorithm...");
        let mut best_threshold = 32;
        let mut best_performance = f64::MAX;
        
        for threshold in [16, 32, 64, 128, 256].iter() {
            let multiplier = StrassenMultiplier::new().with_threshold(*threshold);
            let a = Matrix::random(max_size, max_size);
            let b = Matrix::random(max_size, max_size);
            
            let start = Instant::now();
            let _result = multiplier.multiply(&a, &b);
            let time = start.elapsed().as_secs_f64();
            
            println!("Threshold {}: {:.4}s", threshold, time);
            
            if time < best_performance {
                best_performance = time;
                best_threshold = *threshold;
            }
        }
        
        println!("Optimal threshold: {}", best_threshold);
        best_threshold
    }
    
    pub fn analyze_complexity(sizes: Vec<usize>) {
        println!("Complexity Analysis");
        println!("{:-<40}", "");
        println!("{:<8} {:<12} {:<12} {:<10}", "Size", "Strassen(s)", "Standard(s)", "Ratio");
        
        let multiplier = StrassenMultiplier::new();
        
        for size in sizes {
            let a = Matrix::random(size, size);
            let b = Matrix::random(size, size);
            
            let start = Instant::now();
            let _result = multiplier.multiply(&a, &b);
            let strassen_time = start.elapsed().as_secs_f64();
            
            let start = Instant::now();
            let _result = multiplier.standard_multiply(&a, &b);
            let standard_time = start.elapsed().as_secs_f64();
            
            let ratio = standard_time / strassen_time;
            
            println!("{:<8} {:<12.4} {:<12.4} {:<10.2}", 
                    size, strassen_time, standard_time, ratio);
        }
    }
}

fn main() {
    println!("Advanced Strassen Matrix Multiplication Tool");
    println!("==========================================");
    
    // Example usage
    let multiplier = StrassenMultiplier::new()
        .with_threshold(64)
        .with_parallel(true);
    
    // Create test matrices
    let a = Matrix::random(512, 512);
    let b = Matrix::random(512, 512);
    
    println!("Multiplying 512x512 matrices...");
    let start = Instant::now();
    let result = multiplier.multiply(&a, &b);
    let duration = start.elapsed();
    
    println!("Multiplication completed in: {:?}", duration);
    println!("Result matrix size: {}x{}", result.rows, result.cols);
    
    // Run benchmarks
    println!("\nRunning benchmarks...");
    MatrixBenchmark::compare_algorithms(256, 3);
    
    // Memory analysis
    println!("\nMemory usage analysis:");
    MatrixBenchmark::memory_usage_analysis(1024);
    
    // Find optimal threshold
    println!("\nOptimization analysis:");
    let optimal_threshold = StrassenOptimizer::find_optimal_threshold(512);
    
    // Complexity analysis
    println!("\nComplexity analysis:");
    StrassenOptimizer::analyze_complexity(vec![64, 128, 256, 512]);
}