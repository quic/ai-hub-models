//==============================================================================
//
// Copyright (c) 2018-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef EXECUTABLE_H
#define EXECUTABLE_H 1

#include "graph_status.h"
#include <tuple>
#include <cstdlib>
#include <stdint.h>

class Graph;

#include "weak_linkage.h"
#include "macros_attribute.h"
PUSH_VISIBILITY(default)
namespace hnnx {

typedef volatile uint32_t *const counter_t;
typedef volatile uint32_t *counter_nc_t;

/*
	 * We want to have an abstraction for things that can execute()
	 * so that we can treat them more abstractly than all the things in Op
	 *
	 * This is that interface.
	 *
	 * Note that an important optimization is to be able to obtain the address of the execute call.
	 *
	 * So....
	 * THE "execute()" VIRTUAL FUNCTION MUST BE THE 0th THING IN THE VTABLE
	 * THE "execute()" VIRTUAL FUNCTION MUST NOT CHANGE SIGNATURES
	 *
	 * This allows us to look into the structures to find out more concrete addresses.
	 */
class API_EXPORT Executable { // Almost certainly, don't make this a parent class!
  public:
    using FuncType = GraphStatus (*)(const void *, Graph *);
    using ItemType = std::pair<FuncType, const void *>;
    struct alignas(16) ExecType { // alignment keeps it all in same cache line on hexagon.
        FuncType funcp;
        const void *datap;
        counter_t gate_cp;
        counter_t done_cp;
        ExecType(FuncType const f, const void *const d, counter_t const gc, counter_t const dc)
            : funcp(f), datap(d), gate_cp(gc), done_cp(dc)
        {
        }
    };
    virtual GraphStatus execute(Graph *g) const noexcept = 0; // Needs to be at vtable offset zero!!!
    virtual ItemType compile(Graph &graph_in) const; // Turn this Executable into a function pointer and data pointer.
    virtual ~Executable() = default;
    static const size_t *vtable(Executable const *); // helper function: get vtable
    static const size_t execute_address(Executable const *); // helper function: get address of execute() function

    static GraphStatus no_op_function(const void *, Graph *); // just returns Success.
    static ItemType null_item() { return {no_op_function, nullptr}; }
};

// to execute an Executable::ItemType...

inline GraphStatus execute_item(Graph *graph_in, Executable::ExecType const &itemt)
{
    return (*itemt.funcp)(itemt.datap, graph_in);
}

}; // namespace hnnx

POP_VISIBILITY()

#endif
