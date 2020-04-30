/* SCRATCHPAD.h
 *   by Lut99
 *
 * Created:
 *   30/04/2020, 15:57:41
 * Last edited:
 *   30/04/2020, 16:07:33
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A small class that wraps a large block of memory for scratchpad
 *   allocation. This is the header file.
**/

#ifndef _SCRATCHPAD_H
#define _SCRATCHPAD_H

#include "stddef.h"


/* The Scratchpad struct wraps a block of allocated memory for faster and less OS-dependent dynamic allocation. */
typedef struct SCRATCHPAD {
    size_t size;
    void* data;
} scratchpad;



/***** MEMORY MANAGEMENT *****/

/* Creates a new scratchpad object of given size (in bytes). */
scratchpad* create_scratchpad(size_t bytes);

/* Destroys given scratchpad object. Is equal to free(). */
void destroy_scratchpad(scratchpad* s);

#endif
