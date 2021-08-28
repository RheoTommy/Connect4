from numba import cuda

if __name__ == '__main__':
    cuda.get_current_device().reset()
