from numba import cuda

if __name__ == '__main__':
    device = cuda.get_current_device()
    device.reset()
