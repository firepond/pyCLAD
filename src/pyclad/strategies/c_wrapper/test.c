int simple_threesum(int a, int b, int c) {
    int sum = a + b + c;
    return sum;
}

int sum_function(int num_numbers, int *numbers) {
    int i;
    int sum = 0;
    for (i = 0; i < num_numbers; i++) {
        sum += numbers[i];
    }
    return sum;
}

/*
gcc -fPIC -shared -o libsum.so sum.c

-fPIC: PIC stands for Position Independent Code
       this is required for dynamic linking/shared libraries.
       Means that the library can be executed regardless of its memory address
-shared: produces a shared object that can be linked with others
-o: defines the output file to write to

The steps for calling C functions in python using ctypes!
    - Write your C function(s) (keep in mind how you want to call this in python)
    - Create a shared library (.so file) for your function(s)
    - Using ctypes, create a python wrapper for your C code
    - In python, call the function you want from the wrapper
*/