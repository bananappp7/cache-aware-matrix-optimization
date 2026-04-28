#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

typedef struct Matrix {
    int rows;
    int columns;
    double *data;
} Matrix;

void free_matrix(Matrix* matrix);

Matrix* create_Matrix(int row_number, int column_number) {
    Matrix *new_matrix = (Matrix*) malloc(sizeof(Matrix));
    new_matrix->rows = row_number;
    new_matrix->columns = column_number;
    new_matrix->data = (double*) malloc(row_number * column_number * sizeof(double));
    return new_matrix;
}

Matrix* naive_transposition(Matrix* matrix) {
    Matrix* resulting_matrix = create_Matrix(matrix->columns, matrix->rows);
    for (int row = 0; row < matrix->rows; row ++) {
        for (int column = 0; column < matrix->columns; column ++) { //new_matrix rows
            resulting_matrix->data[row + column * resulting_matrix->columns] = matrix->data[column + row * matrix->columns];
        }
    }
    return resulting_matrix;
}

Matrix* block_transposition(Matrix* matrix) { //do 16x16 matrix blocks
    Matrix* resulting_matrix = create_Matrix(matrix->columns, matrix->rows);
    for (int row = 0; row < matrix->rows/16; row ++) {
        for (int column = 0; column < matrix->columns/16; column ++) { //new_matrix rows
            for (int cc = 0; cc < 16; cc++) {
                for (int rr = 0; rr < 16; rr++){
                    resulting_matrix->data[row * 16 + rr + (column * 16 + cc) *resulting_matrix->columns] = matrix->data[column * 16 + cc + (row * 16 + rr) * matrix->columns];
                }
            }
        }
    }
    return resulting_matrix;
}

Matrix* naive_multiplication(Matrix* first_matrix, Matrix* second_matrix) {
    Matrix *resulting_matrix = create_Matrix(first_matrix->rows, second_matrix->columns);
    for (int rr = 0; rr < resulting_matrix->rows; rr ++) {
        for (int rc = 0; rc < resulting_matrix->columns; rc++) {
            double result = 0;
            for (int value_along = 0; value_along < second_matrix->rows; value_along ++) {
                result += first_matrix->data[value_along + rr * first_matrix->columns] 
                * second_matrix->data[rc + value_along * second_matrix->columns];
            }
            resulting_matrix->data[(rc + rr * resulting_matrix->columns)] = result;
        }
    }
    return resulting_matrix;
}

Matrix* half_brain_multiplication(Matrix* first_matrix, Matrix* second_matrix) {
    Matrix* second_matrix_dummy = naive_transposition(second_matrix);
    Matrix* resulting_matrix = create_Matrix(first_matrix->rows, second_matrix->columns);
    for (int first_row = 0; first_row < first_matrix->rows; first_row ++) {
        for (int second_col = 0; second_col < second_matrix->columns; second_col ++) {
            double result = 0;
            for (int ind = 0; ind < first_matrix->columns; ind ++) {
                result += first_matrix->data[ind + first_row * first_matrix->columns] 
                * second_matrix_dummy->data[ind + second_col * second_matrix_dummy->columns]; 
            }
            resulting_matrix->data[second_col + first_row * resulting_matrix->columns] = result;
        }
    }
    free_matrix(second_matrix_dummy);
    return resulting_matrix;
}

Matrix* blocked_multiplication(Matrix* first_matrix, Matrix* second_matrix) { //USE ONLY FOR 256X256 (or dimensions divisible by 16) FOR NOW
    Matrix* second_matrix_dummy = naive_transposition(second_matrix);
    Matrix* resulting_matrix = create_Matrix(first_matrix->rows, second_matrix->columns);
    for (int i = 0; i < resulting_matrix->rows * resulting_matrix->columns; i++) {
        resulting_matrix->data[i] = 0.0;
    }
    for (int first_row = 0; first_row < (int) first_matrix->rows/16; first_row ++) {
        for (int second_col = 0; second_col < (int) second_matrix->columns/16; second_col ++) {
            for (int k = 0; k < (int) resulting_matrix->rows/16; k++) {
                for (int first_row_mods = 0; first_row_mods < 16; first_row_mods++) {
                    for (int second_col_mods = 0; second_col_mods < 16; second_col_mods ++) {
                        double result = 0;
                        for (int kk = 0; kk < 16; kk ++) {
                            result += first_matrix->data[kk + k * 16+ (first_row *16 + first_row_mods) * first_matrix->columns] 
                            * second_matrix_dummy->data[kk + k * 16 + (second_col * 16 + second_col_mods) * second_matrix_dummy->columns]; 
                        }
                        resulting_matrix->data[(second_col * 16 + second_col_mods) + (first_row * 16 + first_row_mods) * resulting_matrix->columns] += result;
                    }
                }
            }
        }
    }
    free_matrix(second_matrix_dummy);
    return resulting_matrix;
}

Matrix* blocked_multiplication_with_blocked_transposition(Matrix* first_matrix, Matrix* second_matrix) { //USE ONLY FOR 256X256 (or dimensions divisible by 16) FOR NOW
    Matrix* second_matrix_dummy = block_transposition(second_matrix);
    Matrix* resulting_matrix = create_Matrix(first_matrix->rows, second_matrix->columns);
    for (int i = 0; i < resulting_matrix->rows * resulting_matrix->columns; i++) {
        resulting_matrix->data[i] = 0.0;
    }
    for (int first_row = 0; first_row < (int) first_matrix->rows/16; first_row ++) {
        for (int second_col = 0; second_col < (int) second_matrix->columns/16; second_col ++) {
            for (int k = 0; k < (int) resulting_matrix->rows/16; k++) {
                for (int first_row_mods = 0; first_row_mods < 16; first_row_mods++) {
                    for (int second_col_mods = 0; second_col_mods < 16; second_col_mods ++) {
                        double result = 0;
                        for (int kk = 0; kk < 16; kk ++) {
                            result += first_matrix->data[kk + 16 * k + (first_row *16 + first_row_mods) * first_matrix->columns] 
                            * second_matrix_dummy->data[kk + 16 * k + (second_col * 16 + second_col_mods) * second_matrix_dummy->columns]; 
                        }
                        resulting_matrix->data[(second_col * 16 + second_col_mods) + (first_row * 16 + first_row_mods) * resulting_matrix->columns] += result;
                    }
                }
            }
        }
    }
    free_matrix(second_matrix_dummy);
    return resulting_matrix;
}

Matrix* minor_matrix(Matrix* matrix, int element_row, int element_column) {
    int index = 0;
    Matrix* new_matrix = create_Matrix(matrix->rows - 1, matrix->columns - 1);
    for (int row = 0; row < matrix->rows; row++) {
        for (int col = 0; col < matrix->columns; col++) {
            if (row != element_row && col != element_column) {
                new_matrix->data[index] = matrix->data[col + row * matrix->columns];
                index ++;
            }
        }
    }
    return new_matrix;
}

double stupid_determinant(Matrix* matrix) { //square matrix matrix
    if (matrix->rows == 1) {
        return matrix->data[0];
    }
    double output = 0;
    for (int i = 0; i < matrix->columns; i++) {
        Matrix* minormatrix = minor_matrix(matrix, 0, i);
        if (i%2 == 0) {
            output += matrix->data[i] * stupid_determinant(minormatrix);
        }
        else {
            output -= matrix->data[i] * stupid_determinant(minormatrix);
        }
        free_matrix(minormatrix);
       
    }

    return output;
}

Matrix* lower_triangular(Matrix* matrix) {
    Matrix* resulting_matrix = create_Matrix(matrix->rows, matrix->columns);
    for (int index = 0; index < matrix->rows * matrix->columns; index ++) {
        resulting_matrix->data[index] = matrix->data[index];
    }
    for (int col = matrix->columns - 1; col >= 0; col --) {
        for (int row = 0; row < col; row++) {
            double division_result = resulting_matrix->data[col + row * resulting_matrix->columns] / resulting_matrix->data[col + col * resulting_matrix->columns];
            for (int col2 = 0; col2 <= col; col2 ++) {
                resulting_matrix->data[col2 + row * resulting_matrix->columns] = resulting_matrix->data[col2 + row * resulting_matrix->columns] - 
                    division_result * resulting_matrix->data[col2 + col * resulting_matrix->columns];
            }
        }
    }
    return resulting_matrix;
}

Matrix* lower_triangular_purelyd(Matrix* matrix) {
    Matrix* resulting_matrix = create_Matrix(matrix->rows, matrix->columns);
    for (int index = 0; index < matrix->rows * matrix->columns; index ++) {
        resulting_matrix->data[index] = matrix->data[index];
    }
    for (int col = matrix->columns - 1; col >= 0; col --) {
        for (int row = 0; row < col; row++) {
            double division_result = resulting_matrix->data[col + row * resulting_matrix->columns] / resulting_matrix->data[col + col * resulting_matrix->columns];
            for (int col2 = 0; col2 <= col; col2 ++) {
                if (col2 == 0) {
                    resulting_matrix->data[col2] = resulting_matrix->data[col + col * resulting_matrix->columns];
                }
                resulting_matrix->data[col2 + row * resulting_matrix->columns] = resulting_matrix->data[col2 + row * resulting_matrix->columns] - 
                    division_result * resulting_matrix->data[col2 + col * resulting_matrix->columns];
            }
        }
        //resulting_matrix->data[col] = resulting_matrix->data[col + col * resulting_matrix->columns];
    }
    return resulting_matrix;
}

double smart_determinant(Matrix* matrix) {
    Matrix* new_matrix = lower_triangular(matrix);
    double output = 1;

    for (int i = 0; i < new_matrix->rows; i++) {
        output *= new_matrix->data[i + i * new_matrix->columns];
    }
    return output;
}

double smarter_determinant(Matrix* matrix) {
    Matrix* new_matrix = lower_triangular_purelyd(matrix);
    double output = 1;

    for (int i = 0; i < new_matrix->columns; i++) {
        output *= new_matrix->data[i];
    }
    return output;
}


void setRandomData(Matrix* matrix) {
    for (int index = 0; index < matrix->rows * matrix->columns; index ++) {
        matrix->data[index] = (double) (rand() % 8 + 1); //mod 10 placeholder
    }
}

void printMatrix(Matrix* matrix) {
    for (int row = 0; row < matrix->rows; row ++) {
        for (int column = 0; column < matrix->columns; column ++) {
            printf("%lf", matrix->data[(column + row * matrix->columns)]);
            printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

void free_matrix(Matrix* matrix) {
    free(matrix->data);
    free(matrix);
}

int main() {
    srand(time(NULL));
    clock_t start = clock();

    bool multiply = true;
    bool determinant = true;
    int how_smart = 3;

    
    
    if (multiply) {


        
        
        for (int i = 0; i < 100; i++) {
            Matrix* Matrix_A = create_Matrix(256, 256);
            Matrix* Matrix_B = create_Matrix(256, 256);
            setRandomData(Matrix_A);
            setRandomData(Matrix_B);
            Matrix* Matrix_C = create_Matrix(256, 256);
            if (how_smart == 0) {
                Matrix_C = naive_multiplication(Matrix_A, Matrix_B);   
            }
            else if (how_smart == 1) {
                Matrix_C = half_brain_multiplication(Matrix_A, Matrix_B); 
            }
            else if (how_smart == 2) {
                Matrix_C = blocked_multiplication(Matrix_A, Matrix_B);
            }
            else if (how_smart == 3) {
                Matrix_C = blocked_multiplication_with_blocked_transposition(Matrix_A, Matrix_B);
            }
            free_matrix(Matrix_C);
            free_matrix(Matrix_A);
            free_matrix(Matrix_B);
        }
    }

    else if (determinant) {
        for (int i = 0; i < 100; i++) {
            Matrix* Matrix_A = create_Matrix(128, 128);
            setRandomData(Matrix_A);
            if (how_smart == 0) {
                //printf("Son we are NOT doing this")
                double determinant = stupid_determinant(Matrix_A);
                clock_t end3 = clock();
                 double tim3e_taken = (double) (end3 - start) / CLOCKS_PER_SEC;
                printf("One determinant: %f seconds\n", tim3e_taken);
                break;
            }
            else if (how_smart == 1) {
                double determinant = smart_determinant(Matrix_A);
            }
            free_matrix(Matrix_A);
        }
    }


    



    clock_t end = clock();

    double time_taken = (double) (end - start) / CLOCKS_PER_SEC;

    printf("\n");

    if (multiply) {
        if (how_smart == 0) {
            printf("Execution time for 100 naive 256x256 multiplcation: %f seconds\n", time_taken);
        }
        else if (how_smart == 1){
           printf("Execution time for 100 half cache-aware 256x256 multiplication through naive transposition: %f seconds\n", time_taken);
        }
        else if (how_smart == 2) {
            printf("Execution time for 100 half cache-aware 256x256 multiplication through naive transposition with some blocking: %f seconds\n", time_taken);
        }
        else if (how_smart == 3) {
            printf("Execution time for 100 half cache-aware 256x256 multiplication through blocked transposition with some blocking: %f seconds\n", time_taken);
        }
    }
    else if (determinant) {
        if (how_smart == 0) {
            printf("Execution time for 100 naive 128x128 determinant: %f seconds\n", time_taken);
        }
        else if (how_smart == 1) {
            printf("Execution time for 100 more efficient cache-aware 128x128 determinant: %f seconds\n", time_taken);
        }
        else if (how_smart == 2) {
            printf("Execution time for 100 most efficient cache-aware 128x128 determinant: %f seconds\n", time_taken);
        }
    }
    
}
message.txt
