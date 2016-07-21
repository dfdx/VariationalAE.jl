
include("utils.jl")

@runonce type ExH{H}
    head::Symbol
    args::Vector
    typ::Any
end

toExH(ex::Expr) = ExH{ex.head}(ex.head, ex.args, ex.typ)

@runonce type ExNode
    name::Symbol                # name of a variable
    op::Symbol                  # operation that produced it or special symbol
    deps::Vector{Symbol}        # dependencies of this variable (e.g. args of op)
    val::Any                    # value if any (e.g. for consts)
end

@runonce type ExGraph
    tape::Vector{ExNode}        # list of ExNode's
    vars::Dict{Symbol, ExNode}  # map from var name to its node in the graph
    input::Vector{Symbol}       # list of input variables
    last_id::Int                # helper, index of last generated var name
end

function ExGraph(input::Vector{Symbol})
    g = ExGraph(ExNode[], Dict(), input, 0)
    for name in input
        addnode!(g, :input; name=name)
    end
    return g
end

function ExGraph()
    return ExGraph(ExNode[], Dict(), [], 0)
end

function Base.show(io::IO, g::ExGraph)
    print(io, "ExGraph\n")
    for node in g.tape
        print(io, "  $node\n")
    end    
end

function genname(g::ExGraph)
    g.last_id += 1
    return symbol("w$(g.last_id)")
end


## addnode!

function addnode!(g::ExGraph, name::Symbol, op::Symbol,
                  deps::Vector{Symbol}, val::Any)
    node = ExNode(name, op, deps, val)
    push!(g.tape, node)
    g.vars[name] = node
    return name
end

function addnode!(g::ExGraph, op::Symbol;
                  name=:generate, deps=Symbol[], val=nothing)
    name = (name == :generate ? genname(g) : name)
    return addnode!(g, name, op, deps, val)
end


## parse!

"""
Parse Julia expression and build ExGraph in-place.
Return the name of the output variable.
"""
parse!(g::ExGraph, ex::Expr) = parse!(g, toExH(ex))
parse!(g::ExGraph, ::LineNumberNode) = :nil
parse!(g::ExGraph, s::Symbol) = s

function parse!(g::ExGraph, x::Number)
    name = addnode!(g, :constant; val=x)
    return name
end


function parse!(g::ExGraph, ex::ExH{:(=)})
    op = :(=)
    rhs, lhs = ex.args
    name = rhs
    deps = [parse!(g, lhs)]
    addnode!(g, op; name=name, deps=deps)
    return name    
end

function parse!(g::ExGraph, ex::ExH{:call})
    op = ex.args[1]
    # deps = flatten(Symbol, [parse!(g, arg) for arg in ex.args[2:end]])
    deps = Symbol[parse!(g, arg) for arg in ex.args[2:end]]
    name = addnode!(g, op; deps=deps)
    return name
end

function parse!(g::ExGraph, ex::ExH{:block})
    names = Symbol[parse!(g, subex) for subex in ex.args]
    return names[end]
end


################# main ###################

function main()
    ex = quote
        n = x1*x2
        z = n + sin(x1)
    end
    g = ExGraph([:x, :y])
    parse!(g, ex)
end

