/* LINKED LIST.h
 *   by Tim Müller
 *
 * Created:
 *   21/04/2020, 13:01:50
 * Last edited:
 *   4/25/2020, 1:12:40 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file implements a LinkedList datastructure to quickly add
 *   elements. Note that the elements should be pointers, which can
 *   optionally be deallocated as well upon destruction. Additionally, this
 *   library also supports the convertion of LinkedLists to newly allocated
 *   arrays with pointers to the same values. This is the header file.
**/

#ifndef _LINKED_LIST_H
#define _LINKED_LIST_H

#include <stdlib.h>


/***** STRUCTS *****/

/* The struct defining a node within the LinkedList. */
typedef struct LINKEDLIST_NODE {
    llist_node* next;
    void* value;
} llist_node;

/* The struct defining the LinkedList. */
typedef struct LINKEDLIST {
    llist_node* head;
    llist_node* tail;
    size_t size;
} llist;



/***** MEMORY MANAGEMENT *****/

/* Creates an empty LinkedList. */
llist* create_llist();

/* Destroys a given LinkedList object, but does not deallocate the values. */
void destroy_llist(llist* ll);
/* Destroys a given LinkedList object and also deallocates the values. The given function is the function used to free each element. */
void purge_llist(llist* ll, void (*free_func)(void* value));



/***** LINKEDLIST OPERATIONS *****/

/* Appends a new node to the LinkedList. Returns the list to enable chaining. */
llist* llist_append(llist* ll, void* value);

/* Converts the entire list to a fixed-size C array. Note that the object themselves are not actually copied but merely referenced. */
void** llist_toarray(llist* ll);

#endif
