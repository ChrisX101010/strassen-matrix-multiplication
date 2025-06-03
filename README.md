# Advanced Strassen Matrix Multiplication Suite 🚀

An interactive web-based and Python implementation of Strassen's matrix multiplication algorithm with performance visualization and comparison tools.

## 🌟 Features

- Interactive matrix multiplication with multiple algorithms:
  - Strassen's Algorithm
  - Standard Matrix Multiplication
  - Hybrid Approach (adaptive switching)
- Real-time performance visualization
- System resource monitoring
- Automatic threshold optimization
- Memory usage tracking
- Multi-threaded computation support
- Terminal-style system status display

## 📊 Use Cases

1. **Educational Purposes**
   - Demonstrate algorithm complexity differences
   - Visualize performance characteristics
   - Compare multiplication methods

2. **Performance Testing**
   - Benchmark different matrix sizes
   - Find optimal threshold values
   - Measure memory efficiency

3. **Algorithm Research**
   - Compare multiplication strategies
   - Analyze crossover points
   - Study memory-performance tradeoffs

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/ChrisX101010/strassen-matrix-multiplication.git
cd strassen-matrix-multiplication
```

### Web Interface Setup

1. **Using Python's built-in server**:
   ```bash
   python3 -m http.server 8000
   ```
   Then open `http://localhost:8000` in your browser

2. **Using Node.js**:
   ```bash
   # Install http-server globally
   npm install -g http-server
   
   # Run the server
   http-server
   ```
   Visit `http://localhost:8080`

3. **Using VS Code**:
   - Install "Live Server" extension
   - Right-click on `index.html`
   - Select "Open with Live Server"

### Python Implementation Setup

1. **Create Virtual Environment**:
   ```bash
   # Create virtual environment in home directory
   python -m venv ~/strassen_venv

   # Activate virtual environment
   # On Linux/Mac:
   source ~/strassen_venv/bin/activate
   # On Windows:
   # .\strassen_venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   # Install required packages
   pip install numpy matplotlib seaborn psutil
   # Or using requirements file
   pip install -r requirements.txt
   ```

3. **Run Python Script**:
   ```bash
   # Navigate to Python implementation directory
   cd strassen-py

   # Run the script
   python strassen_python.py
   ```

4. **Deactivate Virtual Environment**:
   ```bash
   # When finished
   deactivate
   ```

### Python Script Options
```bash
# Run benchmark with default settings
python strassen_python.py

# Run with specific matrix sizes
python strassen_python.py --sizes 128 256 512

# Run with custom iterations
python strassen_python.py --iterations 5

# Run with visualization
python strassen_python.py --plot
```

## 📖 How to Use

1. **Web Interface**
   - Set matrix size (64-2048)
   - Choose algorithm type
   - Click "Run Single Test"
   - View real-time results

2. **Python Benchmarks**
   - Run full benchmark suite
   - Compare different implementations
   - Generate performance graphs
   - Analyze memory usage

## 🔍 Performance Metrics

The suite measures:
- Execution time
- Operations per second
- Memory usage
- Relative speedup
- Algorithm efficiency

## 💻 System Requirements

**Minimum:**
- 2 CPU cores
- 4GB RAM
- Python 3.8+
- Modern web browser

**Recommended:**
- 4+ CPU cores
- 8GB+ RAM
- Python 3.10+
- Chrome/Firefox latest version

## 🛠 Technical Details

- Web Implementation:
  - Pure JavaScript
  - HTML5 Canvas for visualization
  - Dynamic memory management
  - Web Workers for parallel processing

- Python Implementation:
  - NumPy for matrix operations
  - Multiprocessing for parallelization
  - Matplotlib/Seaborn for visualization
  - psutil for system monitoring

## ⚠️ Limitations

- Browser memory restrictions
- Single-threaded JavaScript constraints
- Maximum matrix size depends on available memory
- Some features Chrome-only (detailed memory stats)

## 🐛 Common Issues & Solutions

1. **Memory Errors**:
   - Reduce matrix size
   - Close other applications
   - Use 64-bit Python

2. **Performance Issues**:
   - Enable parallel processing
   - Adjust threshold values
   - Use optimized Python build

## 📝 License

MIT License - feel free to use and modify as needed!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
