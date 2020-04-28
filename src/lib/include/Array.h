/* ARRAY.h
 *   by Tim Müller
 *
 * Created:
 *   28/04/2020, 19:29:28
 * Last edited:
 *   28/04/2020, 21:30:45
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Defines a very simple array class that enhances usage of basic C
 *   arrays.
**/

#ifndef _ARRAY_H
#define _ARRAY_H

#include "stddef.h"
#include "stdbool.h"


/***** STRUCT DEFINITIONS *****/

/* The Array class aims to improve array operations by providing some useful functions for them and to bundle properties of the array in one convenient object. */
typedef struct ARRAY {
    /* Number of elements in the array. */
    size_t size;
    /* The data stored in the array struct. Note that this usually pointers to the next byte. */
    double* d;
} array;



/***** MEMORY MANAGEMENT *****/

/* Allows the user to allocate a new array object on the stack. */
#define CREATE_STACK_ARRAY(NAME, SIZE) \
    char NAME ## _MEMSPACE[sizeof(array) + SIZE * sizeof(double)]; \
    array* NAME = (array*) NAME ## _MEMSPACE; \
    NAME->size = SIZE; \
    NAME->d = (double*) (NAME ## _MEMSPACE + sizeof(array));

/* Allows the user to allocate a new array object on the stack whos elements are linked to a list of doubles. */
#define CREATE_LINKED_STACK_ARRAY(NAME, SIZE, DATA) \
    char NAME ## _MEMSPACE[sizeof(array)]; \
    array* NAME = (array*) NAME ## _MEMSPACE; \
    NAME->size = SIZE; \
    NAME->d = DATA;


/* Allocates a new array object. */
array* create_empty_array(size_t size);
/* Allocates a new array object and copies given buffer to the internal array. */
array* create_array(size_t size, double* data);
/* Allocates a new array object and sets the internal pointer to the given buffer (so does not copy). */
array* create_linked_array(size_t size, double* data);

/* Copies given list of doubles to the array. Returns the given array object to allow chaining. */
array* fill_array(array* a, const double* data);

/* Copies given (source) array to the first (target) array. Returns the target to allow chaining */
array* copy_array(array* target, const array* source);
/* Copies given array into a new array of equal dimensions. */
array* copy_create_array(const array* a);

/* Destroys given array object. */
void destroy_array(array* a);



/***** USEFUL FUNCTIONS *****/

/* Sums all elements in the given array. */
double array_sum(const array* a);


/***** DEBUG FUNCTIONS *****/

/* Prints an array (without newline) to the given FILE. */
void array_write(FILE* handle, const array* a);
/* Prints an array (with newline) to the given FILE. */
void array_print(FILE* handle, const array* a);

/* Checks if given array is equal to another given array. */
bool array_equals(const array* a1, const array* a2);
/* Checks if given array is equal to a given list if doubles. Note that this not check sizes, so if that is uncertain create an array for the data first and use array_equals. */
bool array_equals2(const array* a, const double* data);


#endif