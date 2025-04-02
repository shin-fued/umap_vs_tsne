import my_tsne
import my_umap
import numpy as np
import time


def benchmark(samples):
    tsne = my_tsne.my_TSNE(0.6)
    umap = my_umap.my_UMAP(0.6)

    # Open file once and write header
    with open("bench_mark.csv", "w") as file:
        file.write("sample_size,tsne_time,umap_time\n")

    for i in range(1, samples):  # No need for step=1
        n = i + 5  # Matrix size

        random_matrix = np.random.rand(n, n)  # Generate random data

        # Benchmark t-SNE
        time_start_tsne = time.time()
        tsne.fit(random_matrix, 30, 2, 200)
        time_end_tsne = time.time()

        # Benchmark UMAP
        time_start_umap = time.time()
        umap.fit(random_matrix, i, 0.1, 2, 200)
        time_end_umap = time.time()

        # Append results to the file
        with open("bench_mark.csv", "a") as file:
            file.write(f"{n * n},{time_end_tsne - time_start_tsne},{time_end_umap - time_start_umap}\n")


benchmark(500)  # Run with 10 sample sizes

