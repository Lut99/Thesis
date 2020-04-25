/* LINKED LIST.c
 *   by Tim Müller
 *
 * Created:
 *   21/04/2020, 13:01:50
 * Last edited:
 *   4/25/2020, 1:12:15 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file implements a LinkedList datastructure to quickly add
 *   elements. Note that the elements should be pointers, which can
 *   optionally be deallocated as well upon destruction. Additionally, this
 *   library also supports the convertion of LinkedLists to newly allocated
 *   arrays with pointers to the same values.
**/

#include <stdio.h>

#include "LinkedList.h"


/***** MEMORY MANAGEMENT *****/

llist* create_llist() {
    llist* to_ret = malloc(sizeof(llist));
    to_ret->head = NULL;
    to_ret->tail = NULL;
    to_ret->size = 0;
    return to_ret;
}

void destroy_llist(llist* ll) {
    // Loop to destroy all nodes
    llist_node* node = ll->head;
    while (node != NULL) {
        llist_node* next = node->next;
        free(node);
        node = next;
    }
    // Destroy the struct itself
    free(ll);
}

void purge_llist(llist* ll, void (*free_func)(void* value)) {
    // Loop to destroy all nodes but make sure to destroy element references as well
    llist_node* node = ll->head;
    while (node != NULL) {
        llist_node* next = node->next;
        (*free_func)(node->value);
        free(node);
        node = next;
    }
    // Destroy the struct itself
    free(ll);
}



/***** LINKEDLIST OPERATIONS *****/

llist* llist_append(llist* ll, void* value) {
    // Create a new node
    llist_node* new_n = malloc(sizeof(llist_node));
    new_n->next = NULL;
    new_n->value = value;

    // Append to the list
    if (ll->head == NULL) {
        // Set as head instead
        ll->head = new_n;
    } else {
        ll->tail->next = new_n;
    }

    // Update the tail & size
    ll->tail = new_n;
    ll->size++;

    // Done
    return ll;
}

void** llist_toarray(llist* ll) {
    // Allocate a new array of values
    void** to_ret = malloc(sizeof(void*) * ll->size);
    
    // Copy all elements
    llist_node* n = ll->head;
    size_t i;
    for (i = 0; i < ll->size; i++) {
        // Sanity check that the list is the size we expect it to be
        if (n == NULL) {
            fprintf(stderr, "ERROR: llist_toarray: reported size of llist (%d) does not match actual size (%d)\n",
                    ll->size, i);
            return NULL;
        }

        // Add the value to the array
        to_ret[i] = n->value;

        // Move to the next in the list
        n = n->next;
    }

    // Sanity check that we reached the end of the list
    if (n != NULL) {
        fprintf(stderr, "ERROR: llist_toarray: reported size of llist (%d) does not match actual size (%d)\n",
                ll->size, i);
        return NULL;
    }

    // Done, return
    return to_ret;
}
