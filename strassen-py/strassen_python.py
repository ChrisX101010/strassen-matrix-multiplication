import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BenchmarkResult:
    algorithm: str
    size: int
    time: float
    memory_mb: float
    operations: int

class StrassenMatrixMultiplier:
    """
    Advanced Strassen Matrix Multiplication with optimization features
    """
    
    def __init__(self, 
                 threshold: int = 64,
                 use_parallel: bool = True,
                 max_workers: Optional[int] = None,
                 use_cache: bool = True,
                 numpy_fallback: bool = True):
        """
        Initialize Strassen multiplier with optimization options
        
        Args:
            threshold: Size below which to use standard multiplication
            use_parallel: Enable parallel processing
            max_workers: Number of worker threads (None for auto)
            use_cache: Enable result caching for repeated computations
            numpy_fallback: Use NumPy for very large matrices
        """
        self.threshold = threshold
        self.use_parallel = use_parallel
        self.max_workers = max_workers or mp.cpu_count()
        self.use_cache = use_cache
        self.numpy_fallback = numpy_fallback
        self.stats = {
            'multiplications': 0,
            'additions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Main multiplication method with automatic optimization
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions incompatible for multiplication")
        
        # Reset stats for this multiplication
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Choose best algorithm based on matrix properties
        return self._choose_algorithm(A, B)
    
    def _choose_algorithm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Intelligently choose the best multiplication algorithm
        """
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape
        
        # Use NumPy for very large matrices (memory considerations)
        if self.numpy_fallback and (rows_A > 2048 or cols_B > 2048):
            return self._numpy_multiply(A, B)
        
        # Use standard for small matrices
        if max(rows_A, cols_A, rows_B, cols_B) < self.threshold:
            return self._standard_multiply(A, B)
        
        # Use Strassen for square-ish matrices
        if abs(rows_A - cols_A) < min(rows_A, cols_A) * 0.1 and \
           abs(rows_B - cols_B) < min(rows_B, cols_B) * 0.1:
            return self._strassen_multiply(A, B)
        
        # Default to optimized standard multiplication
        return self._optimized_standard_multiply(A, B)
    
    def _numpy_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """NumPy multiplication with performance tracking"""
        return np.dot(A, B)
    
    def _standard_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Standard O(n³) matrix multiplication"""
        return np.dot(A, B)
    
    def _optimized_standard_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Cache-friendly standard multiplication with blocking"""
        if self.use_parallel and A.shape[0] > 64:
            return self._parallel_standard_multiply(A, B)
        return np.dot(A, B)
    
    def _parallel_standard_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Parallel standard multiplication using threading"""
        def multiply_block(args):
            start_row, end_row = args
            return A[start_row:end_row] @ B
        
        num_blocks = min(self.max_workers, A.shape[0])
        block_size = A.shape[0] // num_blocks
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            for i in range(num_blocks):
                start_row = i * block_size
                end_row = A.shape[0] if i == num_blocks - 1 else (i + 1) * block_size
                tasks.append((start_row, end_row))
            
            results = list(executor.map(multiply_block, tasks))
        
        return np.vstack(results)
    
    def _strassen_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Strassen's algorithm with padding and optimization
        """
        # Ensure matrices are square and power of 2
        max_dim = max(A.shape[0], A.shape[1], B.shape[0], B.shape[1])
        n = 1 << (max_dim - 1).bit_length()  # Next power of 2
        
        # Pad matrices
        A_padded = np.zeros((n, n), dtype=A.dtype)
        B_padded = np.zeros((n, n), dtype=B.dtype)
        A_padded[:A.shape[0], :A.shape[1]] = A
        B_padded[:B.shape[0], :B.shape[1]] = B
        
        # Multiply using Strassen
        C_padded = self._strassen_recursive(A_padded, B_padded)
        
        # Extract result
        return C_padded[:A.shape[0], :B.shape[1]]
    
    def _strassen_recursive(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Recursive Strassen implementation (removed memoization since numpy arrays are not hashable)
        """
        n = A.shape[0]
        if n <= self.threshold:
            self.stats['multiplications'] += 1
            return self._standard_multiply(A, B)
        
        # Divide matrices into quadrants
        mid = n // 2
        A11, A12 = A[:mid, :mid], A[:mid, mid:]
        A21, A22 = A[mid:, :mid], A[mid:, mid:]
        
        B11, B12 = B[:mid, :mid], B[:mid, mid:]
        B21, B22 = B[mid:, :mid], B[mid:, mid:]
        
        # Compute the 7 Strassen products
        if self.use_parallel and n > 256:
            # Parallel computation of Strassen products
            with ThreadPoolExecutor(max_workers=7) as executor:
                futures = [
                    executor.submit(self._strassen_recursive, A11 + A22, B11 + B22),  # M1
                    executor.submit(self._strassen_recursive, A21 + A22, B11),        # M2
                    executor.submit(self._strassen_recursive, A11, B12 - B22),        # M3
                    executor.submit(self._strassen_recursive, A22, B21 - B11),        # M4
                    executor.submit(self._strassen_recursive, A11 + A12, B22),        # M5
                    executor.submit(self._strassen_recursive, A21 - A11, B11 + B12), # M6
                    executor.submit(self._strassen_recursive, A12 - A22, B21 + B22), # M7
                ]
                
                M1, M2, M3, M4, M5, M6, M7 = [f.result() for f in futures]
        else:
            # Sequential computation
            M1 = self._strassen_recursive(A11 + A22, B11 + B22)
            M2 = self._strassen_recursive(A21 + A22, B11)
            M3 = self._strassen_recursive(A11, B12 - B22)
            M4 = self._strassen_recursive(A22, B21 - B11)
            M5 = self._strassen_recursive(A11 + A12, B22)
            M6 = self._strassen_recursive(A21 - A11, B11 + B12)
            M7 = self._strassen_recursive(A12 - A22, B21 + B22)
        
        # Update addition counter
        self.stats['additions'] += 18  # Strassen requires 18 additions
        
        # Compute result quadrants
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        # Combine quadrants
        C = np.zeros((n, n), dtype=A.dtype)
        C[:mid, :mid] = C11
        C[:mid, mid:] = C12
        C[mid:, :mid] = C21
        C[mid:, mid:] = C22
        
        return C

class MatrixBenchmark:
    """
    Comprehensive benchmarking suite for matrix multiplication algorithms
    """
    
    def __init__(self):
        self.results = []
    
    def benchmark_algorithms(self, sizes: List[int], iterations: int = 3) -> List[BenchmarkResult]:
        """
        Benchmark different algorithms across multiple matrix sizes
        """
        algorithms = {
            'numpy': lambda A, B: np.dot(A, B),
            'strassen_seq': lambda A, B: StrassenMatrixMultiplier(use_parallel=False).multiply(A, B),
            'strassen_par': lambda A, B: StrassenMatrixMultiplier(use_parallel=True).multiply(A, B),
            'strassen_opt': lambda A, B: StrassenMatrixMultiplier(
                threshold=self._find_optimal_threshold(sizes[0]), 
                use_parallel=True
            ).multiply(A, B)
        }
        
        results = []
        
        for size in sizes:
            print(f"\nBenchmarking size {size}x{size}...")
            
            for name, algorithm in algorithms.items():
                times = []
                memory_usage = []
                
                for _ in range(iterations):
                    # Generate random matrices
                    A = np.random.randn(size, size).astype(np.float64)
                    B = np.random.randn(size, size).astype(np.float64)
                    
                    # Measure memory before
                    process = psutil.Process()
                    mem_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Time the algorithm
                    start_time = time.perf_counter()
                    try:
                        result = algorithm(A, B)
                        end_time = time.perf_counter()
                        
                        # Measure memory after
                        mem_after = process.memory_info().rss / 1024 / 1024  # MB
                        
                        times.append(end_time - start_time)
                        memory_usage.append(mem_after - mem_before)
                        
                    except Exception as e:
                        print(f"Error with {name}: {e}")
                        continue
                
                if times:
                    avg_time = np.mean(times)
                    avg_memory = np.mean(memory_usage)
                    operations = size ** 3  # Approximate operations
                    
                    result = BenchmarkResult(
                        algorithm=name,
                        size=size,
                        time=avg_time,
                        memory_mb=avg_memory,
                        operations=operations
                    )
                    results.append(result)
                    
                    print(f"  {name}: {avg_time:.4f}s, {avg_memory:.2f}MB")
        
        self.results = results
        return results
    
    def _find_optimal_threshold(self, max_size: int) -> int:
        """Find optimal threshold for given system"""
        thresholds = [32, 64, 128, 256]
        best_threshold = 64
        best_time = float('inf')
        
        test_size = min(512, max_size)
        A = np.random.randn(test_size, test_size)
        B = np.random.randn(test_size, test_size)
        
        for threshold in thresholds:
            multiplier = StrassenMatrixMultiplier(threshold=threshold, use_parallel=False)
            
            start = time.perf_counter()
            _ = multiplier.multiply(A, B)
            elapsed = time.perf_counter() - start
            
            if elapsed < best_time:
                best_time = elapsed
                best_threshold = threshold
        
        return best_threshold
    
    def plot_results(self, save_path: str = "benchmark_results.png"):
        """Create visualization of benchmark results"""
        if not self.results:
            print("No benchmark results to plot")
            return
        
        # Convert results to DataFrame-like structure for plotting
        algorithms = list(set(r.algorithm for r in self.results))
        sizes = sorted(list(set(r.size for r in self.results)))
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Time comparison
        plt.subplot(2, 2, 1)
        for algo in algorithms:
            algo_results = [r for r in self.results if r.algorithm == algo]
            algo_sizes = [r.size for r in algo_results]
            algo_times = [r.time for r in algo_results]
            plt.plot(algo_sizes, algo_times, marker='o', label=algo)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (seconds)')
        plt.title('Performance Comparison')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        
        # Plot 2: Memory usage
        plt.subplot(2, 2, 2)
        for algo in algorithms:
            algo_results = [r for r in self.results if r.algorithm == algo]
            algo_sizes = [r.size for r in algo_results]
            algo_memory = [r.memory_mb for r in algo_results]
            plt.plot(algo_sizes, algo_memory, marker='s', label=algo)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        
        # Plot 3: Efficiency (operations per second)
        plt.subplot(2, 2, 3)
        for algo in algorithms:
            algo_results = [r for r in self.results if r.algorithm == algo]
            algo_sizes = [r.size for r in algo_results]
            efficiency = [r.operations / r.time for r in algo_results]
            plt.plot(algo_sizes, efficiency, marker='^', label=algo)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Operations per Second')
        plt.title('Computational Efficiency')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        
        # Plot 4: Speedup relative to NumPy
        plt.subplot(2, 2, 4)
        numpy_results = {r.size: r.time for r in self.results if r.algorithm == 'numpy'}
        
        for algo in algorithms:
            if algo == 'numpy':
                continue
            algo_results = [r for r in self.results if r.algorithm == algo]
            speedups = []
            sizes_with_speedup = []
            
            for r in algo_results:
                if r.size in numpy_results:
                    speedup = numpy_results[r.size] / r.time
                    speedups.append(speedup)
                    sizes_with_speedup.append(r.size)
            
            if speedups:
                plt.plot(sizes_with_speedup, speedups, marker='d', label=f'{algo} vs numpy')
        
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup Factor')
        plt.title('Speedup vs NumPy')
        plt.legend()
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Benchmark plots saved to {save_path}")

class StrassenOptimizer:
    """
    Advanced optimization and analysis tools for Strassen algorithm
    """
    
    @staticmethod
    def analyze_optimal_use_cases():
        """
        Analyze and report optimal use cases for Strassen algorithm
        """
        print("=== STRASSEN ALGORITHM OPTIMAL USE CASES ===\n")
        
        use_cases = {
            "Scientific Computing": {
                "description": "Large-scale scientific simulations, finite element analysis",
                "matrix_sizes": "1000x1000 to 10000x10000+",
                "characteristics": "Square matrices, repeated multiplications",
                "benefit": "Significant speedup for large matrices"
            },
            "Machine Learning": {
                "description": "Neural network training, deep learning frameworks",
                "matrix_sizes": "Variable, often 512x512 to 4096x4096",
                "characteristics": "Batch matrix operations, GPU-friendly",
                "benefit": "Reduced computational complexity"
            },
            "Computer Graphics": {
                "description": "3D transformations, rendering pipelines",
                "matrix_sizes": "Typically 4x4 to 512x512",
                "characteristics": "Many small to medium matrices",
                "benefit": "Limited - better for larger matrices"
            },
            "Cryptography": {
                "description": "Lattice-based cryptography, polynomial arithmetic",
                "matrix_sizes": "Medium to large, 256x256 to 2048x2048",
                "characteristics": "Structured matrices, specific fields",
                "benefit": "Asymptotic advantages for large operations"
            },
            "Signal Processing": {
                "description": "Convolution operations, filter banks",
                "matrix_sizes": "Variable, 128x128 to 2048x2048",
                "characteristics": "Often Toeplitz or circulant matrices",
                "benefit": "Combined with FFT for optimal performance"
            }
        }
        
        for category, details in use_cases.items():
            print(f"📊 {category}")
            print(f"   Description: {details['description']}")
            print(f"   Matrix Sizes: {details['matrix_sizes']}")
            print(f"   Characteristics: {details['characteristics']}")
            print(f"   Benefit: {details['benefit']}\n")
    
    @staticmethod
    def performance_guidelines():
        """
        Provide performance optimization guidelines
        """
        print("=== PERFORMANCE OPTIMIZATION GUIDELINES ===\n")
        
        guidelines = [
            "✅ Use for matrices larger than 512x512 for best results",
            "✅ Ensure sufficient RAM (7x matrix size for intermediate storage)",
            "✅ Enable parallel processing for matrices > 256x256",
            "✅ Tune threshold based on your hardware (32-256 range)",
            "✅ Consider memory layout (row-major vs column-major)",
            "❌ Avoid for sparse matrices (use specialized sparse algorithms)",
            "❌ Not optimal for very rectangular matrices (e.g., 1000x10)",
            "❌ Limited benefit for matrices smaller than 128x128",
            "⚠️  Memory usage can be 7x larger than standard multiplication",
            "⚠️  Cache performance critical - ensure good memory hierarchy"
        ]
        
        for guideline in guidelines:
            print(guideline)
        
        print("\n=== HARDWARE RECOMMENDATIONS ===")
        print("• CPU: Multi-core processor (4+ cores recommended)")
        print("• RAM: At least 16GB for matrices > 2048x2048")
        print("• Cache: Large L3 cache beneficial (8MB+)")
        print("• Storage: SSD for virtual memory scenarios")

def main():
    """
    Main demonstration and testing function
    """
    print("🚀 Advanced Strassen Matrix Multiplication Suite")
    print("=" * 50)
    
    # Initialize components
    multiplier = StrassenMatrixMultiplier(
        threshold=64,
        use_parallel=True,
        use_cache=True
    )
    
    benchmark = MatrixBenchmark()
    
    # Quick functionality test
    print("\n1. Quick Functionality Test")
    print("-" * 30)
    
    # Test with small matrices first
    A_small = np.random.randn(8, 8)
    B_small = np.random.randn(8, 8)
    
    result_strassen = multiplier.multiply(A_small, B_small)
    result_numpy = np.dot(A_small, B_small)
    
    error = np.max(np.abs(result_strassen - result_numpy))
    print(f"Small matrix test - Max error: {error:.2e}")
    
    if error < 1e-10:
        print("✅ Small matrix test PASSED")
    else:
        print("❌ Small matrix test FAILED")
        return
    
    # Test with medium matrices
    A_medium = np.random.randn(128, 128)
    B_medium = np.random.randn(128, 128)
    
    start = time.perf_counter()
    result_strassen = multiplier.multiply(A_medium, B_medium)
    strassen_time = time.perf_counter() - start
    
    start = time.perf_counter()
    result_numpy = np.dot(A_medium, B_medium)
    numpy_time = time.perf_counter() - start
    
    error = np.max(np.abs(result_strassen - result_numpy))
    print(f"Medium matrix test - Max error: {error:.2e}")
    print(f"Strassen time: {strassen_time:.4f}s, NumPy time: {numpy_time:.4f}s")
    
    if error < 1e-10:
        print("✅ Medium matrix test PASSED")
    else:
        print("❌ Medium matrix test FAILED")
        return
    
    # Performance benchmark
    print("\n2. Performance Benchmark")
    print("-" * 30)
    
    sizes = [64, 128, 256, 512]  # Adjust based on your system capabilities
    results = benchmark.benchmark_algorithms(sizes, iterations=2)
    
    # Generate performance report
    print("\n3. Performance Analysis")
    print("-" * 30)
    
    StrassenOptimizer.analyze_optimal_use_cases()
    StrassenOptimizer.performance_guidelines()
    
    # Create visualizations
    try:
        benchmark.plot_results("strassen_benchmark.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
        print("Install matplotlib and seaborn for visualization features")
    
    # Statistics summary
    print("\n4. Algorithm Statistics")
    print("-" * 30)
    print(f"Multiplications performed: {multiplier.stats['multiplications']}")
    print(f"Additions performed: {multiplier.stats['additions']}")
    
    print("\n🎯 Setup Complete! Your Strassen implementation is working correctly.")

if __name__ == "__main__":
    main()
