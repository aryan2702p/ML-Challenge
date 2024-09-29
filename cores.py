import multiprocessing

# Get the number of available CPU cores
max_threads = multiprocessing.cpu_count()

print(f"Maximum number of threads (CPU cores) available: {max_threads}")