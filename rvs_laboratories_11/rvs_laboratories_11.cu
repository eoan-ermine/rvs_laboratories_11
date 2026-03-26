// Имеем список (lst) длины n
// Результатом будет его сумма sum = lst[0] + lst[1] + ... + lst[n-1];

#include "wb.h"

#define BLOCK_SIZE 512 //@@ Можно поменять

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total(float *input, float *output, int len) {
  //@@ Загрузка сегмента входного вектора в разделяемую память 

  //@@ Пробег по дереву редукции

  //@@ Запись вычисленной суммы блока в выходной вектор по нужному адресу 
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // Входной одномерный список
  float *hostOutput; // Выходной список
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // количество элементов входного списка 
  int numOutputElements; // количество элементов выходного списка

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Выделение памяти GPU
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Копирование памяти на GPU
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Инициализация размерности блока и сетки
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Запуск ядра
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Копирование памяти обратно с GPU на CPU 
  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Выполните редукцию выходного вектора на хосте
   * ЗАМЕТКА: Операция может быть рекурсивной и поддерживать выходной 
   * вектор любого размера. Для простоты мы не требуем этого в этой 
   * лабораторной.
   ********************************************************************/

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Освободите память GPU
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}