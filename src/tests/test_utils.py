import itertools
import numpy as np


def generate_all_triples(num_entities, num_relations):
    return np.asarray(list(itertools.product(range(num_entities), range(num_relations), range(num_entities))))


def generate_all_queries(triples):
    rhs_pairs = list(dict.fromkeys(map(lambda x: (x[0], x[1]), triples)))
    rel_pairs = list(dict.fromkeys(map(lambda x: (x[0], x[2]), triples)))
    lhs_pairs = list(dict.fromkeys(map(lambda x: (x[1], x[2]), triples)))
    rhs_queries = list(map(lambda x: (x[0], x[1], 0), rhs_pairs))
    rel_queries = list(map(lambda x: (x[0], 0, x[1]), rel_pairs))
    lhs_queries = list(map(lambda x: (0, x[0], x[1]), lhs_pairs))
    return rhs_queries, rel_queries, lhs_queries


def test_generate_all_triples():
    num_entities, num_relations = 100, 10
    triples = generate_all_triples(num_entities, num_relations)
    assert len(triples) == num_entities * num_relations * num_entities
    assert np.all(np.min(triples, axis=0) == (0, 0, 0))
    assert np.all(np.max(triples, axis=0) == (num_entities - 1, num_relations - 1, num_entities - 1))


def test_generate_all_queries():
    num_entities, num_relations = 100, 10
    triples = generate_all_triples(num_entities, num_relations)
    rhs_queries, rel_queries, lhs_queries = generate_all_queries(triples)
    assert len(rhs_queries) == num_entities * num_relations
    assert len(rel_queries) == num_entities * num_entities
    assert len(lhs_queries) == num_entities * num_relations
