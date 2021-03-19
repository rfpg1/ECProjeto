"""Probability models (Chapter 13-15)"""

from collections import defaultdict
from functools import reduce

from agents import Agent
from utils import *
import itertools





# ______________________________________________________________________________


class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, var_name='?', freq=None):
        """If freq is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob = {}
        self.var_name = var_name
        self.values = []
        if freq:
            for (v, p) in freq.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        """Set P(val) = p."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('{}: ' + numfmt).format(v, p) for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.var_name)


class JointProbDist(ProbDist):
    """A discrete probability distribute over a set of variables.
    >>> P = JointProbDist(['X', 'Y']); P[1, 1] = 0.25
    >>> P[1, 1]
    0.25
    >>> P[dict(X=0, Y=1)] = 0.5
    >>> P[dict(X=0, Y=1)]
    0.5"""

    def __init__(self, variables):
        self.prob = {}
        self.variables = variables
        self.vals = defaultdict(list)

    def __getitem__(self, values):
        """Given a tuple or dict of values, return P(values)."""
        values = event_values(values, self.variables)
        return ProbDist.__getitem__(self, values)

    def __setitem__(self, values, p):
        """Set P(values) = p. Values can be a tuple or a dict; it must
        have a value for each of the variables in the joint. Also keep track
        of the values we have seen so far for each variable."""
        values = event_values(values, self.variables)
        self.prob[values] = p
        for var, val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        """Return the set of possible values for a variable."""
        return self.vals[var]

    def __repr__(self):
        return "P({})".format(self.variables)


def event_values(event, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])

    
# ------------------------------  Bayes Node ------------------------------------------

class BayesNode:
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string. cpt, the conditional
        probability table, takes one of these forms:

        * A number, the unconditional probability P(X=true). You can
          use this form when there are no parents.

        * A dict {v: p, ...}, the conditional probability distribution
          P(X=true | parent=v) = p. When there's just one parent.

        * A dict {(v1, v2, ...): p, ...}, the distribution P(X=true |
          parent1=v1, parent2=v2, ...) = p. Each key must have as many
          values as there are parents. You can use this form always;
          the first two are just conveniences.

        In all cases the probability of X being false is left implicit,
        since it follows from P(X=true).

        >>> X = BayesNode('X', '', 0.2)
        >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Z = BayesNode('Z', 'P Q',
        ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        if isinstance(parents, str):
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))
    

# _____________________________________  Bayes Net  _________________________________________


class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs=None):
        """Nodes must be ordered with parents before children."""
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node.variable)

    def variable_node(self, var):
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """Return the domain of var."""
        return [True, False]

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)
    

# ----------------------  calculate the full joint distribution from a Bayes Net

def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    out=JointProbDist(net.variables)
    for row in all_rows(net):
        #print(row)
        produto = 1
        #print(produto)
        for var in net.variables:
            produto *= P_xi_given_parents(var, row, net)
        out[row]=produto
    return out
        
def all_rows(net): return itertools.product(*[[True, False] for var in net.variables])

def P_xi_given_parents_todelete(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]

def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    no_var = net.variable_node(var)
    xi = row[net.variables.index(var)]  # o valor de var na row
    #ev = Evidence(zip(net.variables, row)) # tenho que devolver um dicionário com os pares X=v dos pais
    ev = dict([(p,row[net.variables.index(p)]) for p in no_var.parents])
    return no_var.p(xi,ev)

### como é que dado uma row de valores posso obter um dicionário
### e depois um sub-dicionário

def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result

# ______________________________________________________________________________
    
    


def enumerate_joint_ask(X, e, P):
    """
    [Section 13.3]
    Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P.
    >>> P = JointProbDist(['X', 'Y'])
    >>> P[0,0] = 0.25; P[0,1] = 0.5; P[1,1] = P[2,1] = 0.125
    >>> enumerate_joint_ask('X', dict(Y=1), P).show_approx()
    '0: 0.667, 1: 0.167, 2: 0.167'
    """
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)  # probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e]  # hidden variables.
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
    return Q.normalize()


def enumerate_joint(variables, e, P):
    """Return the sum of those entries in P consistent with e,
    provided variables is P's remaining variables (the ones not in e)."""
    if not variables:
        return P[e]
    Y, rest = variables[0], variables[1:]
    return sum([enumerate_joint(rest, extend(e, Y, y), P) for y in P.values(Y)])


### mostra as várias linhas da conjunta usadas para somar as probabilidades
def enumerate_joint_log(variables, e, P):
    """Return the sum of those entries in P consistent with e,
    provided variables is P's remaining variables (the ones not in e)."""
    if not variables:
        print(e.items(),'=',P[e])
        return P[e]
    Y, rest = variables[0], variables[1:]
    return sum([enumerate_joint_log(rest, extend(e, Y, y), P) for y in P.values(Y)])


# we should not be obliged to give the first input to enumerate_joint...
# that input is obtained from the second input of this higher function
def enumerate_joint_prob(evidencia,conjunta):
    variaveis = [v for v in conjunta.variables if not v in evidencia]
    return enumerate_joint(variaveis,evidencia,conjunta)

#  outputs the condicional of perg given cond, from the joint
#  P(perg|cond) given the joint (conjunta completa)
def prob_cond_calc(perg,cond,conjunta_completa):
    combined = dict(perg, **cond)
    vars_numer = [v for v in conjunta_completa.variables if not v in combined]
    vars_denom = [v for v in conjunta_completa.variables if not v in cond ]
    return enumerate_joint(vars_numer, combined, conjunta_completa) / enumerate_joint(vars_denom, cond, conjunta_completa)



# ------------------------- INDEPENDENCE -----------------------------------

# X is independent (marginally) of Y
def independentes(X,Y,conj_comp):
    """Para todos os valores de X e Y P(X,Y)=P(X)*P(Y)"""
    sets = [conj_comp.values(X),conj_comp.values(Y)]
    enum=list(itertools.product(*sets))
    #enum = enumerate([X,Y],conj_comp)
    return all(prob_cond_calc({X:vs[0],Y:vs[1]},dict(),conj_comp) == prob_cond_calc({X:vs[0]},dict(),conj_comp) * prob_cond_calc({Y:vs[1]}, dict(), conj_comp) for vs in enum)

def condicional_independ(X,Y,cond,conj_comp):
    """X and Y are conditionally independent on cond if and only if P(X|cond)*P(Y|cond)=P(X,Y|Cond)
    for all combinations of the variable's values"""
    sets = [conj_comp.values(x) for x in [X,Y]+cond]
    enum=list(itertools.product(*sets))
    #print(enum)
    c = len(cond)+2
    for e in enum:
        #print('a tratar de',e)
        # form evidence instance zz
        zz=dict()
        for i in range(2,c):
            #print('Var ',cond[i-2],'toma o valor', e[i])
            zz[cond[i-2]]=e[i]
        #print(zz)
        #print(prob_cond_calc({X:e[0],Y:e[1]}, zz, conj_comp))
        #print(prob_cond_calc({X:e[0]},zz, conj_comp) * prob_cond_calc({Y:e[1]},zz, conj_comp))
        #  teste de igualdade ambos arredondados a 5 casas decimais
        if round(prob_cond_calc({X:e[0],Y:e[1]}, zz, conj_comp),5) != round(prob_cond_calc({X:e[0]},zz,conj_comp)*prob_cond_calc({Y:e[1]},zz, conj_comp),5):
            return False
    return True





# ______________________________________________________________________________



