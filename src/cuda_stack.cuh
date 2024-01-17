#ifndef STACK_CUH
#define STACK_CUH

namespace cuda_utils {

    template <typename T, size_t MaxSize = 1024>
    class Stack {
    private:
        T elements[MaxSize];
        size_t size = 0;

    public:
        __host__ __device__ Stack() {
            size = 0;
        }

        __host__ __device__ void push(const T& value) {
            if (size == MaxSize) {
                printf("Stack is full -- overflow error\n");
            } else {
                elements[size++] = value;
            }
        }

        __host__ __device__ T pop() {
            if (size == 0) {
                printf("Stack is empty -- underflow error\n");
                return elements[0];
            } else {
                return elements[--size];
            }
        }

        __host__ __device__ bool is_empty() const {
            return size == 0;
        }

        __host__ __device__ size_t get_size() const {
            return size;
        }
    };

}
#endif // STACK_CUH
