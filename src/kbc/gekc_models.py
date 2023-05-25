import abc
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Union, Any, Mapping

import torch
from torch import nn
from ogb import linkproppred

from kbc.utils import safelog
from kbc.models import KBCModel, filtering
from kbc.distributions import init_params_, Categorical, TwinCategorical

TSRL_MODELS_NAMES = [
    'NNegCP', 'NNegRESCAL', 'NNegTuckER', 'NNegComplEx',
    'SquaredCP', 'SquaredComplEx', 'TypedSquaredCP', 'TypedSquaredComplEx'
]


class TractableKBCModel(KBCModel):
    def __init__(
        self,
        sizes: Tuple[int, int, int],
        rank: int,
        rank_r: Optional[int] = None,
        role_entity: bool = False,
        init_dist: str = 'normal',
        init_loc: float = 0.0,
        init_scale: float = 1e-1,
        base_dist: str = 'categorical'
    ):
        """
        Tractable KBC Circuit.

        :param sizes: Tuple of number of entities and number of relation types.
        :param rank: The rank of the decomposition, or simply the number of channels.
        :param rank_r: The specific rank for the relations. If None then rank will be used.
        :param role_entity: Whether to duplicate entities embeddings based on their role, i.e. subject or object.
        :param init_dist: The distribution to use to initialize the parameters.
        :param init_loc: Initial location for embeddings initialization.
        :param init_scale: Initial scale for embeddings initialization.
        :param base_dist: The base distributions layer to use for entities' embeddings.
        """
        super(TractableKBCModel, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.rank_r = rank if rank_r is None else rank_r
        self.role_entity = role_entity

        # Initialize the batch size of the entity input distribution
        self.batch_size = self.rank
        if self.role_entity:
            self.batch_size *= 2

        # Initialize the entities' embeddings
        if base_dist == 'categorical':
            self.ent_embeddings = Categorical(
                self.sizes[0], self.batch_size,
                init_dist=init_dist, init_loc=init_loc, init_scale=init_scale
            )
            if self.role_entity:
                init_params_(self.ent_embeddings.logits.data[:, :self.rank], init_dist, init_loc, init_scale)
                init_params_(self.ent_embeddings.logits.data[:, self.rank:], init_dist, init_loc, init_scale)
        elif base_dist == 'twin-categorical':
            self.ent_embeddings = TwinCategorical(
                self.sizes[0], self.batch_size,
                init_dist=init_dist, init_loc=init_loc, init_scale=init_scale
            )
            if self.role_entity:
                init_params_(self.ent_embeddings.logits.data[:, :self.rank], init_dist, init_loc, init_scale)
                init_params_(self.ent_embeddings.logits.data[:, self.rank:], init_dist, init_loc, init_scale)
        elif base_dist == 'embedding':
            self.ent_embeddings = nn.Embedding(self.sizes[0], self.batch_size)
            if self.role_entity:
                init_params_(self.ent_embeddings.weight.data[:, :self.rank], init_dist, init_loc, init_scale)
                init_params_(self.ent_embeddings.weight.data[:, self.rank:], init_dist, init_loc, init_scale)
            else:
                init_params_(self.ent_embeddings.weight.data, init_dist, init_loc, init_scale)
        else:
            raise ValueError("Unknown base distributions layer named {}".format(base_dist))

        # Initialize the relations' embeddings
        if base_dist == 'categorical':
            self.rel_embeddings = Categorical(
                self.sizes[1], self.rank_r,
                init_dist=init_dist, init_loc=init_loc, init_scale=init_scale
            )
        elif base_dist == 'twin-categorical':
            self.rel_embeddings = TwinCategorical(
                self.sizes[1], self.rank_r,
                init_dist=init_dist, init_loc=init_loc, init_scale=init_scale
            )
        elif base_dist == 'embedding':
            self.rel_embeddings = nn.Embedding(self.sizes[1], self.rank_r)
            init_params_(self.rel_embeddings.weight.data, init_dist, init_loc, init_scale)
        else:
            raise ValueError("Unknown base distributions layer named {}".format(base_dist))

    @abc.abstractmethod
    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit bottom-up.

        :param lhs: A batch of subject embeddings of shape (batch, rank).
        :param rel: A batch of relation embeddings of shape (batch, rank_r).
        :param rhs: A batch of object embeddings of shape (batch, rank).
        :return: The result of the circuit of shape (batch, 1).
        """
        pass

    @abc.abstractmethod
    def eval_circuit_partial(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor, target: str = 'rhs'):
        """
        Perform partial circuit computations given entities and relation embeddings.
        This function is useful for computing rankings efficiently and to perform circuit marginalization.

        :param lhs: A batch of subject embeddings of shape (batch, rank).
        :param rel: A batch of relation embeddings of shape (batch, rank).
        :param rhs: A batch of object embeddings of shape (batch, rank).
        :param target: The target component to rank of the triple.
        :return: Partial circuit computation results of shape (batch, rank).
        """
        pass

    @abc.abstractmethod
    def eval_circuit_all(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        """
        Perform circuit computations over several candidates.
        This function is useful for computing rankings efficiently.

        :param qs: A batch of partial circuit computations of shape (batch, rank).
        :param cs: Entities or relations candidated embeddings of shape (n, rank).
        :return: All circuit computation results of shape (batch, n).
        """
        pass

    def eval_circuit_all_ogb(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        """
        Perform circuit computations over several candidates (for ogb datasets only).
        This function is useful for computing rankings efficiently.

        :param qs: A batch of partial circuit computations of shape (batch, rank).
        :param cs: Entities candidated embeddings of shape (n, num_cands, rank).
        :return: All circuit computation results of shape (batch, num_cands, n).
        """
        pass

    @abc.abstractmethod
    def entities_partition_function(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the partition function of the entities input distribution.

        :return: The partition function for both subjects and objects.
        """
        pass

    @abc.abstractmethod
    def relations_partition_function(self) -> torch.Tensor:
        """
        Compute the partition function of the relations input distribution.

        :return: The partition function for relation types.
        """
        pass

    @abc.abstractmethod
    def partition_function(self) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Return the partition function (log-space).

        :return: The partition function (log-space), and a tuple of the partition function of input distributions.
        """
        pass

    @abc.abstractmethod
    def log_likelihood(
            self,
            x: torch.Tensor,
            con_rhs: bool = False,
            con_rel: bool = False,
            con_lhs: bool = False,
            return_ll: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute the log-likelihoods of a batch of triples.
        Using three flags it is possible to compute conditional log-likelihoods of a batch of triples.

        :param x: A batch of triples of shape (batch, 3).
        :param con_rhs: Whether to condition w.r.t. the subjects.
        :param con_rel: Whether to condition w.r.t. the relations.
        :param con_lhs: Whether to condition w.r.t. the objects.
        :param return_ll: Whether to return the log-likelihoods of the triples.
        :return: The optionally None log-likelihoods with shape (batch, 1).
            A tuple of conditional log-likelihoods is returned, based on the con_(*) flags.
        """
        pass

    @torch.no_grad()
    @abc.abstractmethod
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample facts (triples) from the encoded probability distribution.

        :param n_samples: The number of samples.
        :return: The samples of shape (n_samples, 3).
        """
        pass

    def index_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the embeddings of subjects, relations and objects in a batch of triples.

        :param x: A batch of triples of shape (batch, 3).
        :return: The embeddings of subjects, relations and objects.
        """
        # Get the embeddings of the input triples
        lhs = self.ent_embeddings(x[:, 0])
        rel = self.rel_embeddings(x[:, 1])
        rhs = self.ent_embeddings(x[:, 2])
        if self.role_entity:
            lhs, rhs = lhs[:, :self.rank], rhs[:, self.rank:]
        return lhs, rel, rhs

    def forward(
            self,
            x: torch.Tensor,
            score_rhs: bool = False,
            score_rel: bool = False,
            score_lhs: bool = False,
            score_ll: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Evaluate the circuit in forward mode.
        Notice that the scores in this case are actually (conditional) log-likelihoods.

        :param x: A batch of triples of shape (batch, 3).
        :param score_rhs: Whether to condition w.r.t. the objects.
        :param score_rel: Whether to condition w.r.t. the relations.
        :param score_lhs: Whether to condition w.r.t. the subjects.
        :param score_ll: Whether to compute the log-likelihoods of the triples.
        :return: The log-likelihoods and the conditional log-likelihoods (as a list).
        """
        # Compute the log-likelihoods
        return self.log_likelihood(
            x, con_rhs=score_rhs, con_rel=score_rel, con_lhs=score_lhs, return_ll=score_ll
        )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the scores of a batch of triples.

        :param x: A batch of triples of shape (batch, 3).
        :return: The scores with shape (batch, 1).
        """
        # Evaluate the circuit
        lhs, rel, rhs = self.index_embeddings(x)
        return self.eval_circuit(lhs, rel, rhs)

    def get_candidates(
            self,
            chunk_begin: int,
            chunk_size: int,
            target: str = 'rhs',
            indices: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get the candidate embeddings.
        This function is useful for computing rankings efficiently.

        :param chunk_begin: The index of the chunk of candidates embeddings to extract.
        :param chunk_size: The number of candidates embeddings to extract.
        :param target: The target w.r.t. compute the rankings.
        :param indices: The specific indices of the candidates.
        :param device: The device to use.
        :return: The candidate embeddings.
        """
        if target == 'rhs' or target == 'lhs':
            if indices is None:
                idx = torch.arange(chunk_begin, chunk_begin + chunk_size, device=device)
                cs = self.ent_embeddings(idx)
                if self.role_entity:
                    if target == 'rhs':
                        cs = cs[:, self.rank:]
                    else:
                        cs = cs[:, :self.rank]
            else:
                bsz = indices.shape[0]
                num_cands = indices.shape[1]
                if target == 'rhs':
                    idx = indices[:, num_cands // 2:]
                else:
                    idx = indices[:, :num_cands // 2]
                cs = self.ent_embeddings(idx.reshape(-1))
                if self.role_entity:
                    if target == 'rhs':
                        cs = cs[:, self.rank:]
                    else:
                        cs = cs[:, :self.rank]
                if len(cs.shape) > 2:
                    cs = cs.reshape(bsz, num_cands // 2, -1, cs.shape[2])
                else:
                    cs = cs.reshape(bsz, num_cands // 2, -1)
        elif target == 'rel':
            idx = torch.arange(chunk_begin, chunk_begin + chunk_size, device=device)
            cs = self.rel_embeddings(idx)
        else:
            raise ValueError("Invalid target value")
        return cs

    def get_queries(self, x: torch.Tensor, target: str = 'rhs') -> torch.Tensor:
        """
        Get partial scores computations given a batch of triples and a target.
        This function is useful for computing rankings efficiently.

        :param x: A batch of triples of shape (batch, 3).
        :param target: The target w.r.t. compute the rankings.
        :return: Partial scores computations of shape (batch, rank).
        """
        # Evaluate the circuit partially based on the target flag
        lhs, rel, rhs = self.index_embeddings(x)
        return self.eval_circuit_partial(lhs, rel, rhs, target=target)

    def get_ranking(
        self,
        queries: torch.Tensor,
        filters: Dict[Tuple[int, int], List[int]],
        batch_size: int = 1000,
        chunk_size: int = -1,
        candidates: str = 'rhs'
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Returns filtered ranking for each query.

        :param queries: A torch.LongTensor of triples (lhs, rel, rhs).
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking.
        :param batch_size: Maximum number of queries processed at once.
        :param chunk_size: Maximum number of answering candidates processed at once
        :param candidates: The candidates for ranking. At the moment it can be either 'lhs' or 'rhs'.
        :return: The ranks and the MAP entity predictions.
        """
        if candidates not in ['lhs', 'rhs']:
            raise NotImplementedError("get_ranking() not implemented for non-entity candidates")

        query_type = candidates
        if chunk_size < 0:  # not chunking, score against all candidates at once
            chunk_size = self.sizes[2]  # entity ranking
        ranks = torch.ones(len(queries))
        predicted = torch.zeros(len(queries))
        diverged = False

        with torch.no_grad():
            # Loop for batches over entities
            c_begin = 0
            while c_begin < self.sizes[2]:
                # Get the candidates embeddings
                cs = self.get_candidates(c_begin, chunk_size, target=query_type, device=queries.device)

                # Loop for batches over queries
                b_begin = 0
                while b_begin < len(queries):
                    # Get the queries, i.e. partial score computations for (s, p, .) or (., p, o)
                    these_queries = queries[b_begin:b_begin + batch_size]
                    qs = self.get_queries(these_queries, target=query_type)

                    # Compute the scores and the target scores (in log-domain)
                    scores = self.eval_circuit_all(qs, cs)
                    targets = self.score(these_queries)

                    # Check for -Infs or NaNs
                    if not diverged:
                        if not torch.all(torch.isfinite(scores)) or not torch.all(torch.isfinite(targets)):
                            diverged = True

                    # Compute the ranks and the MAP entity
                    if filters is not None:
                        scores = filtering(scores, these_queries, filters,
                                           n_rel=self.sizes[1], n_ent=self.sizes[2],
                                           c_begin=c_begin, chunk_size=chunk_size,
                                           query_type=query_type)
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    predicted[b_begin:b_begin + batch_size] = torch.max(scores, dim=1)[1].cpu()
                    b_begin += batch_size
                c_begin += chunk_size
        return ranks, predicted, diverged

    def get_metrics_ogb(
            self,
            queries: torch.Tensor,
            evaluator: linkproppred.Evaluator,
            batch_size: int = 1000,
            query_type: str = 'rhs',
    ) -> Tuple[dict, bool]:
        """
        No need to filter since the provided negatives are ready filtered.

        :param queries: a torch.LongTensor of triples (lhs, rel, rhs).
        :param evaluator: The ogbl evaluator.
        :param batch_size: maximum number of queries processed at once.
        :param query_type: The query type, either rhs or lhs.
        :return: The metrics and whether it has diverged (for these models it is always false)
        """
        diverged = False
        test_logs = defaultdict(list)
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                if these_queries.shape[1] > 5:  # more than h,r,t,h_type,t_type
                    tot_neg = 1000 if evaluator.name in ['ogbl-biokg', 'ogbl-wikikg2'] else 0
                    neg_indices = these_queries[:, 3:3+tot_neg]
                    chunk_begin, chunk_size = None, None
                else:
                    neg_indices = None
                    chunk_begin, chunk_size = 0, self.sizes[2]  # all the entities
                q = self.get_queries(these_queries, target=query_type)
                cands = self.get_candidates(
                    chunk_begin, chunk_size,
                    target=query_type, indices=neg_indices, device=these_queries.device
                )
                if isinstance(q, tuple):
                    q, cands = q[0], cands[0]
                if cands.dim() >= (4 if isinstance(self, NNegComplEx) else 3):
                    # Each example has a different negative candidate embedding matrix
                    scores = self.eval_circuit_all_ogb(q, cands)
                else:
                    scores = self.eval_circuit_all(q, cands)
                targets = self.score(these_queries)  # positive scores
                # Check for -Infs or NaNs
                if not diverged:
                    if not torch.all(torch.isfinite(scores)) or not torch.all(torch.isfinite(targets)):
                        diverged = True
                batch_results = evaluator.eval({'y_pred_pos': targets.squeeze(-1), 'y_pred_neg': scores})
                del targets, scores, q, cands
                for metric in batch_results:
                    test_logs[metric].append(batch_results[metric])
                b_begin += batch_size
        metrics = {}
        for metric in test_logs:
            metrics[metric] = torch.cat(test_logs[metric]).mean().item()
        return metrics, diverged


class NNegKBCModel(TractableKBCModel):
    def __init__(
        self,
        sizes: Tuple[int, int, int],
        rank: int,
        rank_r: Optional[int] = None,
        role_entity: bool = False,
        init_dist: str = 'normal',
        init_loc: float = 0.0,
        init_scale: float = 1e-3,
        base_dist: str = 'categorical'
    ):
        super(NNegKBCModel, self).__init__(
            sizes=sizes, rank=rank, rank_r=rank_r,
            role_entity=role_entity, init_dist=init_dist,
            init_loc=init_loc, init_scale=init_scale,
            base_dist=base_dist
        )

    def filter_inverted_relations(self):
        prev_rel_logits = self.rel_embeddings.logits.data
        n_relations, rank = prev_rel_logits.shape[0] // 2, prev_rel_logits.shape[1]
        self.rel_embeddings = Categorical(n_relations, rank)
        self.rel_embeddings.logits.data.copy_(prev_rel_logits[:n_relations])

    def eval_circuit_all(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        m_q, _ = torch.max(qs, dim=1, keepdim=True)
        m_c, _ = torch.max(cs, dim=1, keepdim=True)
        qs = torch.exp(qs - m_q)
        cs = torch.exp(cs - m_c)
        x = torch.mm(qs, cs.t())
        x = safelog(x) + m_q + m_c.t()
        return x

    def eval_circuit_all_ogb(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        qs = qs.unsqueeze(-1)  # (b, d, 1)
        # cs (b, k, d)
        m_q, _ = torch.max(qs, dim=1, keepdim=True)  # (b, 1, 1)
        m_c, _ = torch.max(cs, dim=2, keepdim=True)  # (b, k, 1)
        qs = torch.exp(qs - m_q)
        cs = torch.exp(cs - m_c)
        x = torch.bmm(cs, qs)  # (b, k, 1)
        x = safelog(x) + m_q + m_c
        x = x.squeeze(-1)  # (b, k)
        return x

    def entities_partition_function(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z_ent = self.ent_embeddings.partition_function()
        if self.role_entity:
            z_lhs_ent, z_rhs_ent = z_ent[:, :self.rank], z_ent[:, self.rank:]
            return z_lhs_ent, z_rhs_ent
        return z_ent, z_ent

    def relations_partition_function(self) -> torch.Tensor:
        z_rel = self.rel_embeddings.partition_function()
        return z_rel

    def partition_function(self) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Compute the partition function of input distributions
        z_lhs_ent, z_rhs_ent = self.entities_partition_function()
        z_rel = self.relations_partition_function()

        # Evaluate the circuit to compute the partition function
        z = self.eval_circuit(z_lhs_ent, z_rel, z_rhs_ent)
        return z, (z_lhs_ent, z_rel, z_rhs_ent)

    def log_likelihood(
            self,
            x: torch.Tensor,
            con_rhs: bool = False,
            con_rel: bool = False,
            con_lhs: bool = False,
            return_ll: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Get the embeddings of the input triples
        lhs, rel, rhs = self.index_embeddings(x)

        # Evaluate the circuit
        log_scores = self.eval_circuit(lhs, rel, rhs)

        # Compute the partition function and the log-probabilities, if specified
        if return_ll:
            z, (z_lhs_ent, z_rel, z_rhs_ent) = self.partition_function()
            log_probs = log_scores - z
        else:
            log_probs = None
            # Get the partition function of input distributions
            if con_lhs or con_rhs:
                z_lhs_ent, z_rhs_ent = self.entities_partition_function()
            else:
                z_lhs_ent = z_rhs_ent = None
            if con_rel:
                z_rel = self.relations_partition_function()
            else:
                z_rel = None

        # Perform CON inference
        con_log_probs = [None, None, None]
        if con_rhs:
            con_log_probs[0] = log_scores - self.eval_circuit(lhs, rel, z_rhs_ent)
        if con_rel:
            con_log_probs[1] = log_scores - self.eval_circuit(lhs, z_rel, rhs)
        if con_lhs:
            con_log_probs[2] = log_scores - self.eval_circuit(z_lhs_ent, rel, rhs)

        return log_probs, con_log_probs

    @torch.no_grad()
    def sample_from_indices(self, lhs_idx: torch.Tensor, rel_idx: torch.Tensor, rhs_idx: torch.Tensor) -> torch.Tensor:
        lhs_idx = self.ent_embeddings.sample(lhs_idx)
        rel_idx = self.rel_embeddings.sample(rel_idx)
        rhs_idx = self.ent_embeddings.sample(rhs_idx)
        return torch.stack([lhs_idx, rel_idx, rhs_idx], dim=1)


class SquaredKBCModel(TractableKBCModel):
    def __init__(
        self,
        sizes: Tuple[int, int, int],
        rank: int,
        rank_r: Optional[int] = None,
        role_entity: bool = False,
        init_dist: str = 'log-normal',
        init_loc: float = 0.0,
        init_scale: float = 1e-3
    ):
        super(SquaredKBCModel, self).__init__(
            sizes=sizes, rank=rank, rank_r=rank_r,
            role_entity=role_entity, init_dist=init_dist,
            init_loc=init_loc, init_scale=init_scale,
            base_dist='embedding'
        )

        # Initialize epsilon values to avoid log-zeroing
        self.eps = 1.0 / (sizes[0] * sizes[1] * sizes[2])
        self.e_eps = 1.0 / (sizes[1] * sizes[2])
        self.r_eps = 1.0 / (sizes[0] * sizes[2])

    def filter_inverted_relations(self):
        prev_rel_embeddings = self.rel_embeddings.weight.data
        n_relations, rank = prev_rel_embeddings.shape[0] // 2, prev_rel_embeddings.shape[1]
        self.rel_embeddings = nn.Embedding(n_relations, rank)
        self.rel_embeddings.weight.data.copy_(prev_rel_embeddings[:n_relations])

    def partition_function(self) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Compute the partition function of input distributions
        z_lhs_ent, z_rhs_ent = self.entities_partition_function()
        z_rel = self.relations_partition_function()

        # Evaluate the circuit
        z = self.eval_circuit_partition_function(z_lhs_ent, z_rel, z_rhs_ent)
        return z, (z_lhs_ent, z_rel, z_rhs_ent)

    @abc.abstractmethod
    def marginalized_score(
            self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor,
            z_lhs_ent: torch.Tensor, z_rel: torch.Tensor, z_rhs_ent: torch.Tensor, target: str = 'rhs'
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def eval_circuit_partition_function(self, z_lhs: Any, z_rel: Any, z_rhs: Any) -> torch.Tensor:
        """
        Compute the partition function of the overall circuit, given the partition functions of the input functionals.

        :param z_lhs: The parititon function of the subject input functionals.
        :param z_rel: The parititon function of the relation input functionals.
        :param z_rhs: The parititon function of the object input functionals.
        :return: The partition function of the overall circuit.
        """
        pass

    def eval_circuit_all(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.mm(qs, cs.t()))

    def eval_circuit_all_ogb(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.bmm(cs, qs.unsqueeze(-1)).squeeze(-1))

    def log_likelihood(
            self,
            x: torch.Tensor,
            con_rhs: bool = False,
            con_rel: bool = False,
            con_lhs: bool = False,
            return_ll: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Get the embeddings of the input triples
        lhs, rel, rhs = self.index_embeddings(x)

        # Evaluate the circuit
        scores = self.eval_circuit(lhs, rel, rhs)
        log_scores = torch.log(scores + self.eps)

        # Compute the partition function and the log-probabilities, if specified
        if return_ll:
            z, (z_lhs_ent, z_rel, z_rhs_ent) = self.partition_function()
            log_probs = log_scores - torch.log1p(z)
        else:
            log_probs = None
            # Get the partition function of input distributions
            if con_lhs or con_rhs:
                z_lhs_ent, z_rhs_ent = self.entities_partition_function()
            else:
                z_lhs_ent = z_rhs_ent = None
            if con_rel:
                z_rel = self.relations_partition_function()
            else:
                z_rel = None

        # Perform CON inference
        log_con_probs = [None, None, None]
        if con_rhs:
            mar_rhs = self.marginalized_score(lhs, rel, rhs, z_lhs_ent, z_rel, z_rhs_ent, target='rhs')
            log_con_probs[0] = log_scores - torch.log(mar_rhs + self.e_eps)
        if con_rel:
            mar_rel = self.marginalized_score(lhs, rel, rhs, z_lhs_ent, z_rel, z_rhs_ent, target='rel')
            log_con_probs[1] = log_scores - torch.log(mar_rel + self.r_eps)
        if con_lhs:
            mar_lhs = self.marginalized_score(lhs, rel, rhs, z_lhs_ent, z_rel, z_rhs_ent, target='lhs')
            log_con_probs[2] = log_scores - torch.log(mar_lhs + self.e_eps)

        return log_probs, log_con_probs


class NNegCP(NNegKBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, **kwargs):
        """
        NNegtonic Canonical Polyadic (CP) decomosition Circuit (or CP+).

        :param sizes: Tuple of number of entities and number of relation types.
        :param rank: The rank of the decomposition, or simply the number of channels.
        :param kwargs: Additional parameters to pass to the base class.
        """
        super(NNegCP, self).__init__(sizes, rank, role_entity=True, init_dist='exp-dirichlet', **kwargs)

    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        x = lhs + rel + rhs
        return torch.logsumexp(x, dim=1, keepdim=True)

    def eval_circuit_partial(
            self,
            lhs: torch.Tensor,
            rel: torch.Tensor,
            rhs: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            x = lhs + rel
        elif target == 'rel':
            x = lhs + rhs
        elif target == 'lhs':
            x = rel + rhs
        else:
            raise ValueError("Invalid target value")
        return x

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        branch = torch.randint(self.rank, size=(n_samples,))
        return self.sample_from_indices(branch, branch, branch + self.rank)


class NNegRESCAL(NNegKBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, **kwargs):
        """
        NNegtonic RESCAL Circuit (or RESCAL+).

        :param sizes: Tuple of number of entities and number of relation types.
        :param rank: The rank of the decomposition, or simply the number of channels.
        :param kwargs: Additional parameters to pass to the base class.
        """
        super(NNegRESCAL, self).__init__(sizes, rank, rank_r=rank * rank, init_dist='exp-dirichlet', **kwargs)

    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        # Subtract maximum for numerical stability
        m_lhs, _ = torch.max(lhs, dim=1, keepdim=True)
        m_rel, _ = torch.max(rel, dim=1, keepdim=True)
        m_rhs, _ = torch.max(rhs, dim=1, keepdim=True)
        lhs = torch.exp(lhs - m_lhs)
        rel = torch.exp(rel - m_rel)
        rhs = torch.exp(rhs - m_rhs)
        rel = rel.view(-1, self.rank, self.rank)

        # Log-Einsum-Exp trick
        x = torch.einsum('bi,bij,bj->b', lhs, rel, rhs)
        x = torch.unsqueeze(x, dim=1)
        x = torch.log(x) + m_lhs + m_rel + m_rhs
        return x

    def eval_circuit_partial(
            self,
            lhs: torch.Tensor,
            rel: torch.Tensor,
            rhs: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            m_lhs, _ = torch.max(lhs, dim=1, keepdim=True)
            m_rel, _ = torch.max(rel, dim=1, keepdim=True)
            lhs = torch.exp(lhs - m_lhs)
            rel = torch.exp(rel - m_rel)
            rel = rel.view(-1, self.rank, self.rank)
            x = torch.einsum('bi,bij->bj', lhs, rel)
            x = torch.log(x) + m_lhs + m_rel
        elif target == 'rel':
            m_lhs, _ = torch.max(lhs, dim=1, keepdim=True)
            m_rhs, _ = torch.max(rhs, dim=1, keepdim=True)
            lhs = torch.exp(lhs - m_lhs)
            rhs = torch.exp(rhs - m_rhs)
            x = torch.einsum('bi,bj->bij', lhs, rhs)
            x = torch.log(x.view(-1, self.rank_r)) + m_lhs + m_rhs
        elif target == 'lhs':
            m_rhs, _ = torch.max(rhs, dim=1, keepdim=True)
            m_rel, _ = torch.max(rel, dim=1, keepdim=True)
            rhs = torch.exp(rhs - m_rhs)
            rel = torch.exp(rel - m_rel)
            rel = rel.view(-1, self.rank, self.rank)
            x = torch.einsum('bj,bij->bi', rhs, rel)
            x = torch.log(x) + m_rhs + m_rel
        else:
            raise ValueError("Invalid target value")
        return x

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        branches = torch.randint(self.rank, size=(n_samples, 2))
        lhs_branch, rhs_branch = branches[:, 0], branches[:, 1]
        return self.sample_from_indices(lhs_branch, lhs_branch * self.rank + rhs_branch, rhs_branch)


class NNegTuckER(NNegKBCModel):
    def __init__(
            self,
            sizes: Tuple[int, int, int],
            rank: int,
            rank_r: int,
            init_scale: float = 1e-3,
            **kwargs
    ):
        """
        NNegtonic TuckER Circuit (or TuckER+).

        :param sizes: Tuple of number of entities and number of relation types.
        :param rank: The rank of the decomposition, or simply the number of channels.
        :param rank_r: The specific rank for the relations.
        :param init_scale: Initial scale for embeddings initialization.
        :param kwargs: Additional parameters to pass to the base class.
        """
        super(NNegTuckER, self).__init__(
            sizes, rank, rank_r=rank_r, init_scale=init_scale, init_dist='exp-dirichlet', **kwargs
        )

        # Initialize the core tensor parameter
        core = torch.empty(self.rank * self.rank * self.rank_r)
        init_params_(core, 'exp-dirichlet', init_scale=init_scale)
        self.core = nn.Parameter(core, requires_grad=True)

    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        # Subtract maximum for numerical stability
        m_core, _ = torch.max(self.core, dim=0)
        m_lhs, _ = torch.max(lhs, dim=1, keepdim=True)
        m_rel, _ = torch.max(rel, dim=1, keepdim=True)
        m_rhs, _ = torch.max(rhs, dim=1, keepdim=True)
        core = torch.exp(self.core - m_core)
        lhs = torch.exp(lhs - m_lhs)
        rel = torch.exp(rel - m_rel)
        rhs = torch.exp(rhs - m_rhs)
        core = core.view(self.rank, self.rank, self.rank_r)

        # Log-Einsum-Exp trick
        x = torch.einsum('ijk,bi,bj,bk->b', core, lhs, rhs, rel)
        x = torch.unsqueeze(x, dim=1)
        x = torch.log(x) + m_core + m_lhs + m_rel + m_rhs
        return x

    def eval_circuit_partial(
            self,
            lhs: torch.Tensor,
            rel: torch.Tensor,
            rhs: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        m_core, _ = torch.max(self.core, dim=0)
        core = torch.exp(self.core - m_core)
        core = core.view(self.rank, self.rank, self.rank_r)
        if target == 'rhs':
            m_lhs, _ = torch.max(lhs, dim=1, keepdim=True)
            m_rel, _ = torch.max(rel, dim=1, keepdim=True)
            lhs = torch.exp(lhs - m_lhs)
            rel = torch.exp(rel - m_rel)
            x = torch.einsum('ijk,bi,bk->bj', core, lhs, rel)
            x = torch.log(x) + m_core + m_lhs + m_rel
        elif target == 'rel':
            m_lhs, _ = torch.max(lhs, dim=1, keepdim=True)
            m_rhs, _ = torch.max(rhs, dim=1, keepdim=True)
            lhs = torch.exp(lhs - m_lhs)
            rhs = torch.exp(rhs - m_rhs)
            x = torch.einsum('ijk,bi,bj->bk', core, lhs, rhs)
            x = torch.log(x) + m_core + m_lhs + m_rhs
        elif target == 'lhs':
            m_rhs, _ = torch.max(rhs, dim=1, keepdim=True)
            m_rel, _ = torch.max(rel, dim=1, keepdim=True)
            rhs = torch.exp(rhs - m_rhs)
            rel = torch.exp(rel - m_rel)
            x = torch.einsum('ijk,bj,bk->bi', core, rhs, rel)
            x = torch.log(x) + m_core + m_rhs + m_rel
        else:
            raise ValueError("Invalid value")
        return x

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        probs = torch.exp(self.core - torch.logsumexp(self.core, dim=0))
        branch = torch.multinomial(probs, n_samples, replacement=True)
        lhs_branch, rhs_branch, rel_branch = (
            torch.div(branch, self.rank_r * self.rank, rounding_mode='floor') % self.rank,
            torch.div(branch, self.rank_r, rounding_mode='floor') % self.rank,
            branch % self.rank_r
        )
        return self.sample_from_indices(lhs_branch, rel_branch, rhs_branch)


class NNegComplEx(NNegKBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, **kwargs):
        """
        NNegtonic ComplEx Circuit (or ComplEx+).
        The parameters of ComplEx are reparametrized (the real part or the imaginary part) in order to achieve
        non-negative outputs.

        :param sizes: Tuple of number of entities and number of relation types.
        :param rank: The rank of the decomposition, or simply the number of channels.
        :param kwargs: Additional parameters to pass to the base class.
        """
        super(NNegComplEx, self).__init__(
            sizes, rank,
            base_dist='twin-categorical', init_dist='exp-dirichlet', **kwargs
        )
        self.register_buffer('lhs_idx', torch.tensor([0, 1, 0, 1]))
        self.register_buffer('rel_idx', torch.tensor([0, 0, 1, 1]))
        self.register_buffer('rhs_idx', torch.tensor([0, 1, 1, 0]))

    def filter_inverted_relations(self):
        prev_rel_logits = self.rel_embeddings.logits.data
        prev_rel_weight = self.rel_embeddings.weight.data
        n_relations, rank = prev_rel_logits.shape[0] // 2, prev_rel_logits.shape[1]
        self.rel_embeddings = TwinCategorical(n_relations, rank)
        self.rel_embeddings.logits.data.copy_(prev_rel_logits[:n_relations])
        self.rel_embeddings.weight.data.copy_(prev_rel_weight[:n_relations])

    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        # Compute the four components of the ComplEx scoring function
        lhs = lhs[:, :, self.lhs_idx]  # (batch, rank, 4)
        rel = rel[:, :, self.rel_idx]  # (batch, rank, 4)
        rhs = rhs[:, :, self.rhs_idx]  # (batch, rank, 4)
        x = lhs + rel + rhs
        x = torch.logsumexp(x, dim=1)  # (batch, 4)

        # Subtract maximum for numerical stability
        m_x, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.exp(x - m_x)

        # Combine as in the ComplEx scoring function
        x = safelog(x[:, 0] + x[:, 1] + x[:, 2] - x[:, 3])
        return torch.unsqueeze(x, dim=1) + m_x

    def eval_circuit_partial(
            self,
            lhs: torch.Tensor,
            rel: torch.Tensor,
            rhs: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            lhs = lhs[:, :, self.lhs_idx]
            rel = rel[:, :, self.rel_idx]
            x = lhs + rel
        elif target == 'rel':
            lhs = lhs[:, :, self.lhs_idx]
            rhs = rhs[:, :, self.rhs_idx]
            x = lhs + rhs
        elif target == 'lhs':
            rel = rel[:, :, self.rel_idx]
            rhs = rhs[:, :, self.rhs_idx]
            x = rel + rhs
        else:
            raise ValueError("Invalid target value")
        return x

    def get_candidates(
            self,
            chunk_begin: int,
            chunk_size: int,
            target: str = 'rhs',
            indices: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:
        cs = super().get_candidates(chunk_begin, chunk_size, target=target, indices=indices, device=device)
        if target == 'rhs':
            cs = cs[..., self.rhs_idx]
        elif target == 'rel':
            cs = cs[..., self.rel_idx]
        elif target == 'lhs':
            cs = cs[..., self.lhs_idx]
        else:
            raise ValueError("Invalid target value")
        return cs

    def eval_circuit_all(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        m_q, _ = torch.max(qs, dim=1, keepdim=True)
        m_c, _ = torch.max(cs, dim=1, keepdim=True)
        qs = torch.exp(qs - m_q)
        cs = torch.exp(cs - m_c)
        x = torch.einsum('bim,nim->bnm', qs, cs)
        x = safelog(x) + m_q + torch.permute(m_c, (1, 0, 2))
        m_x, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.exp(x - m_x)
        x = x[:, :, 0] + x[:, :, 1] + x[:, :, 2] - x[:, :, 3]
        x = safelog(x) + m_x.squeeze(dim=2)
        return x

    def eval_circuit_all_ogb(self, qs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        # qs (b, d, 4)
        # cs (b, d, 4)
        m_q, _ = torch.max(qs, dim=1, keepdim=True)  # (b, 1, 4)
        m_c, _ = torch.max(cs, dim=2, keepdim=True)  # (b, k, 1, 4)
        qs = torch.exp(qs - m_q)
        cs = torch.exp(cs - m_c)
        x = torch.einsum('bkim,bim->bkm', cs, qs)
        x = safelog(x) + m_q + m_c.squeeze(2)
        m_x, _ = torch.max(x, dim=2, keepdim=True)  # (b, k, 1)
        x = torch.exp(x - m_x)
        x = x[:, :, 0] + x[:, :, 1] + x[:, :, 2] - x[:, :, 3]
        x = safelog(x) + m_x.squeeze(dim=2)  # (b, k)
        return x

    @torch.no_grad()
    def sample_from_indices(
            self,
            lhs_idx: Tuple[torch.Tensor, torch.Tensor],
            rel_idx: Tuple[torch.Tensor, torch.Tensor],
            rhs_idx: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        lhs_idx = self.ent_embeddings.twin_sample(lhs_idx[0], lhs_idx[1])
        rel_idx = self.rel_embeddings.twin_sample(rel_idx[0], rel_idx[1])
        rhs_idx = self.ent_embeddings.twin_sample(rhs_idx[0], rhs_idx[1])
        return torch.stack([lhs_idx, rel_idx, rhs_idx], dim=1)

    @torch.no_grad()
    def approximate_sample(self, n_samples: int = 1) -> torch.Tensor:
        samples = torch.empty(0, 3, dtype=torch.long, device=self.lhs_idx.device)
        # Perform rejection sampling (as in TwinSPNs)
        while len(samples) < n_samples:
            # Sample one of the three branches of positive PCs
            pos_branch = torch.randint(3, size=(n_samples,), device=self.lhs_idx.device)
            # Sample the second branch of the selected positive PC
            batch_idx = torch.randint(self.rank, size=(n_samples,), device=self.lhs_idx.device)
            # Sample triples from the selected positive PCs, which are actually NNegCP
            sig_idx = self.lhs_idx[pos_branch], self.rel_idx[pos_branch], self.rhs_idx[pos_branch]
            x = self.sample_from_indices((sig_idx[0], batch_idx), (sig_idx[1], batch_idx), (sig_idx[2], batch_idx))
            # Compute the scores given by the positive PC only
            # This can be done by evaluating the overall PC and subtracting the score given by the negative PC
            lhs_emb = self.ent_embeddings(x[:, 0])[:, :self.rank, 1]
            rel_emb = self.rel_embeddings(x[:, 1])[:, :, 1]
            rhs_emb = self.ent_embeddings(x[:, 2])[:, :self.rank, 0]
            neg_scores = torch.exp(torch.logsumexp(lhs_emb + rel_emb + rhs_emb, dim=1))
            overall_scores = torch.exp(self.score(x).squeeze(dim=1))
            pos_scores = overall_scores - neg_scores  # (b)
            # Sample y from [0, pos_scores) and accept the samples only if y < overall_scores
            y = pos_scores * torch.rand_like(pos_scores)
            mask = y < overall_scores
            samples = torch.cat([samples, x[mask]], dim=0)  # Concatenate the accepted samples
        # Drop the rest of the samples
        return samples[:n_samples]

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        # Exact sampling via inverse transform sampling, just like squared circuits
        # s ~ Pr(S)
        z_lhs_ent, z_rhs_ent = self.entities_partition_function()  # (1, d, 2)
        z_rel = self.relations_partition_function()
        idx = torch.arange(0, self.sizes[0], device=self.ent_embeddings.weight.device)
        lhs_embs = self.ent_embeddings(idx)  # (n_e, d, 2)
        lhs_scores = self.eval_circuit(lhs_embs, z_rel, z_rhs_ent).squeeze(dim=1)  # (n_e)
        lhs_probs = torch.exp(lhs_scores - torch.logsumexp(lhs_scores, dim=0, keepdim=True))  # (n_e)
        assert torch.allclose(torch.sum(lhs_probs, dim=0), torch.tensor(1.0))
        lhs_idx = torch.multinomial(lhs_probs, n_samples, replacement=True)  # (n_samples)
        lhs = lhs_embs[lhs_idx]  # (n_samples, d, 2)

        # r ~ Pr(P | s)
        idx = torch.arange(0, self.sizes[1], device=self.rel_embeddings.weight.device)
        rel_embs = self.rel_embeddings(idx)  # (n_r, d, 2)
        rel_scores = list()
        for i in range(n_samples):
            rel_scores.append(self.eval_circuit(lhs[i].unsqueeze(dim=0), rel_embs, z_rhs_ent).squeeze(dim=1))  # (n_r)
        rel_scores = torch.stack(rel_scores)  # (n_samples, n_r)
        rel_probs = torch.exp(rel_scores - torch.logsumexp(rel_scores, dim=1, keepdim=True))  # (n_samples, n_r)
        assert torch.allclose(torch.sum(rel_probs, dim=1), torch.tensor(1.0))
        rel_idx = torch.multinomial(rel_probs, 1).squeeze(dim=1)  # (n_samples)
        rel = rel_embs[rel_idx]  # (n_samples, d, 2)

        # o ~ Pr(O | s, r)
        idx = torch.arange(0, self.sizes[2], device=self.ent_embeddings.weight.device)
        rhs_embs = self.ent_embeddings(idx)  # (n_e, d, 2)
        rhs_scores = list()
        for i in range(n_samples):
            rhs_scores.append(self.eval_circuit(lhs[i].unsqueeze(dim=0), rel[i].unsqueeze(dim=0), rhs_embs).squeeze(dim=1))  # (n_r)
        rhs_scores = torch.stack(rhs_scores)  # (n_samples, n_e)
        rhs_probs = torch.exp(rhs_scores - torch.logsumexp(rhs_scores, dim=1, keepdim=True))  # (n_samples, n_e)
        assert torch.allclose(torch.sum(rhs_probs, dim=1), torch.tensor(1.0))
        rhs_idx = torch.multinomial(rhs_probs, 1).squeeze(dim=1)  # (n_samples)
        return torch.stack([lhs_idx, rel_idx, rhs_idx], dim=1)


class SquaredCP(SquaredKBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, **kwargs):
        super(SquaredCP, self).__init__(sizes, rank, role_entity=True, init_dist='centered-cp-log-normal', **kwargs)

    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        x = lhs * rel * rhs
        x = torch.sum(x, dim=1, keepdim=True)
        return torch.square(x)

    def eval_circuit_partial(
            self,
            lhs: torch.Tensor,
            rel: torch.Tensor,
            rhs: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            x = lhs * rel
        elif target == 'rel':
            x = lhs * rhs
        elif target == 'lhs':
            x = rel * rhs
        else:
            raise ValueError("Invalid target value")
        return x

    def entities_partition_function(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lhs_ent = self.ent_embeddings.weight[:, :self.rank]
        rhs_ent = self.ent_embeddings.weight[:, self.rank:]
        z_lhs_ent = torch.mm(lhs_ent.t(), lhs_ent)
        z_rhs_ent = torch.mm(rhs_ent.t(), rhs_ent)
        return z_lhs_ent, z_rhs_ent

    def relations_partition_function(self) -> torch.Tensor:
        rel = self.rel_embeddings.weight
        z_rel = torch.mm(rel.t(), rel)
        return z_rel

    def eval_circuit_partition_function(self, z_lhs: torch.Tensor, z_rel: torch.Tensor, z_rhs: torch.Tensor) \
            -> torch.Tensor:
        # Evaluate the circuit to compute the partition function
        z = z_lhs * z_rel * z_rhs
        return torch.sum(z.view(-1, self.rank * self.rank), dim=1, keepdim=True)

    def marginalized_score(
            self,
            lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor,
            z_lhs_ent: torch.Tensor, z_rel: torch.Tensor, z_rhs_ent: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            p_rhs = lhs * rel
            mar = torch.sum(p_rhs * torch.mm(p_rhs, z_rhs_ent), dim=1, keepdim=True)
        elif target == 'rel':
            p_rel = lhs * rhs
            mar = torch.sum(p_rel * torch.mm(p_rel, z_rel), dim=1, keepdim=True)
        elif target == 'lhs':
            p_lhs = rel * rhs
            mar = torch.sum(p_lhs * torch.mm(p_lhs, z_lhs_ent), dim=1, keepdim=True)
        else:
            raise ValueError("Invalid target value")
        return mar

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        # Perform sampling by inverse transform sampling
        #
        lhs_embs = self.ent_embeddings.weight[:, :self.rank]
        rel_embs = self.rel_embeddings.weight
        rhs_embs = self.ent_embeddings.weight[:, self.rank:]
        z_rhs_ent = torch.mm(rhs_embs.t(), rhs_embs)
        z_rel = torch.mm(rel_embs.t(), rel_embs)

        # s ~ \Pr(S) = (\sum_{P,O} \phi(S,P,O)) / (\sum_{S',P,O} \phi(S',P,O))
        lhs_scores = torch.sum(lhs_embs * torch.mm(lhs_embs, z_rel * z_rhs_ent), dim=1)        # (n_e)
        lhs_scores = lhs_scores + self.e_eps * self.sizes[1]                                   # (n_e)
        lhs_probs = lhs_scores * torch.reciprocal(torch.sum(lhs_scores, dim=0, keepdim=True))  # (n_e)
        lhs_idx = torch.multinomial(lhs_probs, n_samples, replacement=True)                    # (n_samples)
        lhs = lhs_embs[lhs_idx]                                                                # (n_samples, d)
        assert torch.isclose(torch.sum(lhs_probs), torch.tensor(1.0))

        # p ~ \Pr(P | S=s) = (\sum_{O} \phi(s,P,O)) / (\sum_{P',O} \phi(s,P',O))
        p_rel = rel_embs.unsqueeze(dim=0) * lhs.unsqueeze(dim=1)                               # (n_samples, n_r, d)
        rel_scores = torch.sum(p_rel * torch.einsum('kni,ij->knj', p_rel, z_rhs_ent), dim=2)   # (n_samples, n_r)
        rel_scores = rel_scores + self.e_eps                                                   # (n_samples, n_r)
        rel_probs = rel_scores * torch.reciprocal(torch.sum(rel_scores, dim=1, keepdim=True))  # (n_samples, n_r)
        rel_idx = torch.multinomial(rel_probs, 1).squeeze(dim=1)                               # (n_samples)
        rel = rel_embs[rel_idx]                                                                # (n_samples, d)
        assert torch.allclose(torch.sum(rel_probs, dim=1), torch.tensor(1.0))

        # o ~ \Pr(O | S=s,P=p) = \phi(s,p,O) / (\sum_{O'} \phi(s,p,O'))
        p_rhs = lhs * rel                                                                      # (n_samples, d)
        rhs_scores = torch.square(torch.mm(p_rhs, rhs_embs.t()))                               # (n_samples, n_e)
        rhs_scores = rhs_scores + self.eps                                                     # (n_samples, n_e)
        rhs_probs = rhs_scores * torch.reciprocal(torch.sum(rhs_scores, dim=1, keepdim=True))  # (n_samples, n_e)
        rhs_idx = torch.multinomial(rhs_probs, 1).squeeze(dim=1)                               # (n_samples)
        assert torch.allclose(torch.sum(rhs_probs, dim=1), torch.tensor(1.0))

        return torch.stack([lhs_idx, rel_idx, rhs_idx], dim=1)


class SquaredComplEx(SquaredKBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, **kwargs):
        super(SquaredComplEx, self).__init__(sizes, rank * 2, init_dist='centered-complex-log-normal', **kwargs)
        self.ri_rank = rank

    def index_embeddings(self, x: torch.Tensor) \
            -> Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor]
            ]:
        # Get the embeddings of the input triples
        lhs = self.ent_embeddings(x[:, 0])
        rel = self.rel_embeddings(x[:, 1])
        rhs = self.ent_embeddings(x[:, 2])
        lhs = lhs[:, :self.ri_rank], lhs[:, self.ri_rank:]
        rel = rel[:, :self.ri_rank], rel[:, self.ri_rank:]
        rhs = rhs[:, :self.ri_rank], rhs[:, self.ri_rank:]
        return lhs, rel, rhs

    def eval_circuit(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        x0 = lhs[0] * rel[0]
        x1 = lhs[1] * rel[0]
        x2 = lhs[0] * rel[1]
        x3 = lhs[1] * rel[1]
        x = torch.sum((x0 - x3) * rhs[0] + (x1 + x2) * rhs[1], dim=1, keepdim=True)
        return torch.square(x)

    def eval_circuit_partial(
            self,
            lhs: torch.Tensor,
            rel: torch.Tensor,
            rhs: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            x = torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        elif target == 'lhs':
            x = torch.cat([
                rhs[0] * rel[0] + rhs[1] * rel[1],
                rhs[1] * rel[0] - rhs[0] * rel[1]
            ], 1)
        elif target == 'rel':
            x = torch.cat([
                lhs[0] * rhs[0] + lhs[1] * rhs[1],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]
            ], 1)
        else:
            raise ValueError("Invalid target value")
        return x

    def entities_partition_function(self) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        re_ent, im_ent = self.ent_embeddings.weight[:, :self.ri_rank], self.ent_embeddings.weight[:, self.ri_rank:]
        re_re_ent = torch.mm(re_ent.t(), re_ent)  # (d, d)
        re_im_ent = torch.mm(re_ent.t(), im_ent)  # (d, d)
        im_im_ent = torch.mm(im_ent.t(), im_ent)  # (d, d)
        z_ent = (re_re_ent, re_im_ent, im_im_ent)
        return z_ent, z_ent

    def relations_partition_function(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        re_rel, im_rel = self.rel_embeddings.weight[:, :self.ri_rank], self.rel_embeddings.weight[:, self.ri_rank:]
        re_re_rel = torch.mm(re_rel.t(), re_rel)  # (d, d)
        re_im_rel = torch.mm(re_rel.t(), im_rel)  # (d, d)
        im_im_rel = torch.mm(im_rel.t(), im_rel)  # (d, d)
        z_rel = (re_re_rel, re_im_rel, im_im_rel)
        return z_rel

    def eval_circuit_partition_function(
            self,
            z_lhs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            z_rel: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            z_rhs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # z_lhs and z_rhs are the same tensor in ComplEx (and also the same instance)
        (re_re_ent, re_im_ent, im_im_ent) = z_lhs  # or z_rel
        (re_re_rel, re_im_rel, im_im_rel) = z_rel
        im_re_ent = re_im_ent.t()
        # ComplEx^2 = (A + B + C - D)^2
        #           = A^2 + B^2 + C^2 + D^2 + AB + BA + AC + CA - AD - DA + BC + CB - BD - DB - CD - DC
        # where A := rrr
        #       B := iri
        #       C := rii
        #       D := iir
        x0 = re_re_ent * re_re_rel
        x1 = im_im_ent * re_re_rel
        x2 = re_re_ent * im_im_rel
        x3 = im_im_ent * im_im_rel
        sq = (x0 + x3) * re_re_ent + (x1 + x2) * im_im_ent  # A^2 + B^2 + C^2 + D^2
        # AB + AB^T + AC + AC^T - AD - AD^T + BC + BC^T - BD - BD^T - CD - CD^T
        # = (AB - CD) + (AB - CD)^T  ... after HEAVY optimizations and cancellations
        v1 = re_re_rel * re_im_ent  # AB
        v2 = im_im_rel * im_re_ent  # CD
        co = (v1 - v2) * re_im_ent
        z = sq + 2.0 * co
        return torch.sum(z.view(-1, self.ri_rank * self.ri_rank), dim=1, keepdim=True)

    def marginalized_score(
            self,
            lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor,
            z_lhs_ent: torch.Tensor, z_rel: torch.Tensor, z_rhs_ent: torch.Tensor,
            target: str = 'rhs'
    ) -> torch.Tensor:
        if target == 'rhs':
            (re_re_ent, re_im_ent, im_im_ent) = z_rhs_ent
            p0_rhs = lhs[0] * rel[0]
            p1_rhs = lhs[1] * rel[0]
            p2_rhs = lhs[0] * rel[1]
            p3_rhs = lhs[1] * rel[1]
            u0 = torch.mm(p0_rhs, re_re_ent)
            u1 = torch.mm(p1_rhs, im_im_ent)
            u2 = torch.mm(p2_rhs, im_im_ent)
            u3 = torch.mm(p3_rhs, re_re_ent)
            z0 = torch.sum(p0_rhs * u0, dim=1, keepdim=True)
            z1 = torch.sum(p1_rhs * u1, dim=1, keepdim=True)
            z2 = torch.sum(p2_rhs * u2, dim=1, keepdim=True)
            z3 = torch.sum(p3_rhs * u3, dim=1, keepdim=True)
            v0 = torch.sum(p3_rhs * u0, dim=1, keepdim=True)
            v1 = torch.sum(p2_rhs * u1, dim=1, keepdim=True)
            v2 = torch.sum((p1_rhs + p2_rhs) * torch.mm(p0_rhs - p3_rhs, re_im_ent), dim=1, keepdim=True)
            mar = z0 + z1 + z2 + z3 + 2.0 * (v2 + v1 - v0)
        elif target == 'rel':
            (re_re_rel, re_im_rel, im_im_rel) = z_rel
            p0_rel = lhs[0] * rhs[0]
            p1_rel = lhs[1] * rhs[0]
            p2_rel = lhs[0] * rhs[1]
            p3_rel = lhs[1] * rhs[1]
            u0 = torch.mm(p0_rel, re_re_rel)
            u1 = torch.mm(p1_rel, im_im_rel)
            u2 = torch.mm(p2_rel, im_im_rel)
            u3 = torch.mm(p3_rel, re_re_rel)
            z0 = torch.sum(p0_rel * u0, dim=1, keepdim=True)
            z1 = torch.sum(p1_rel * u1, dim=1, keepdim=True)
            z2 = torch.sum(p2_rel * u2, dim=1, keepdim=True)
            z3 = torch.sum(p3_rel * u3, dim=1, keepdim=True)
            v0 = torch.sum(p3_rel * u0, dim=1, keepdim=True)
            v1 = torch.sum(p2_rel * u1, dim=1, keepdim=True)
            v2 = torch.sum((p2_rel - p1_rel) * torch.mm(p0_rel + p3_rel, re_im_rel), dim=1, keepdim=True)
            mar = z0 + z1 + z2 + z3 + 2.0 * (v2 + v0 - v1)
        elif target == 'lhs':
            re_re_ent, re_im_ent, im_im_ent = z_lhs_ent
            p0_lhs = rhs[0] * rel[0]
            p1_lhs = rhs[1] * rel[0]
            p2_lhs = rhs[0] * rel[1]
            p3_lhs = rhs[1] * rel[1]
            u0 = torch.mm(p0_lhs, re_re_ent)
            u1 = torch.mm(p1_lhs, im_im_ent)
            u2 = torch.mm(p2_lhs, im_im_ent)
            u3 = torch.mm(p3_lhs, re_re_ent)
            z0 = torch.sum(p0_lhs * u0, dim=1, keepdim=True)
            z1 = torch.sum(p1_lhs * u1, dim=1, keepdim=True)
            z3 = torch.sum(p2_lhs * u2, dim=1, keepdim=True)
            z2 = torch.sum(p3_lhs * u3, dim=1, keepdim=True)
            v0 = torch.sum(p3_lhs * u0, dim=1, keepdim=True)
            v1 = torch.sum(p2_lhs * u1, dim=1, keepdim=True)
            v2 = torch.sum((p1_lhs - p2_lhs) * torch.mm((p0_lhs + p3_lhs), re_im_ent), dim=1, keepdim=True)
            mar = z0 + z1 + z2 + z3 + 2.0 * (v2 + v0 - v1)
        else:
            raise ValueError("Invalid target value")
        return mar

    @torch.no_grad()
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        # Perform sampling by inverse transform sampling
        #
        re_ent_embs = self.ent_embeddings.weight[:, :self.ri_rank]
        im_ent_embs = self.ent_embeddings.weight[:, self.ri_rank:]
        re_rel_embs = self.rel_embeddings.weight[:, :self.ri_rank]
        im_rel_embs = self.rel_embeddings.weight[:, self.ri_rank:]
        (re_re_ent, re_im_ent, im_im_ent), _ = self.entities_partition_function()
        re_re_rel, re_im_rel, im_im_rel = self.relations_partition_function()

        # s ~ \Pr(S) = (\sum_{P,O} \phi(S,P,O)) / (\sum_{S',P,O} \phi(S',P,O))
        u0 = torch.mm(re_ent_embs, re_re_rel * re_re_ent)  # (n_e, d)
        u1 = torch.mm(im_ent_embs, re_re_rel * im_im_ent)  # (n_e, d)
        u2 = torch.mm(re_ent_embs, im_im_rel * im_im_ent)  # (n_e, d)
        u3 = torch.mm(im_ent_embs, im_im_rel * re_re_ent)  # (n_e, d)
        u4 = torch.mm(im_ent_embs, re_re_rel * re_im_ent)  # (n_e, d)
        u5 = torch.mm(re_ent_embs, im_im_rel * re_im_ent)  # (n_e, d)
        v0 = torch.sum(re_ent_embs * (u0 + u2), dim=1)     # (n_e)
        v1 = torch.sum(im_ent_embs * (u1 + u3), dim=1)     # (n_e)
        v2 = torch.sum(re_ent_embs * u4, dim=1)            # (n_e)
        v3 = torch.sum(im_ent_embs * u5, dim=1)            # (n_e)
        lhs_scores = v0 + v1 + v2 - v3                     # (n_e)
        lhs_scores = lhs_scores + self.e_eps * self.sizes[1]
        lhs_probs = lhs_scores * torch.reciprocal(torch.sum(lhs_scores, dim=0, keepdim=True))  # (n_e)
        lhs_idx = torch.multinomial(lhs_probs, n_samples, replacement=True)                    # (n_samples)
        re_lhs, im_lhs = re_ent_embs[lhs_idx],  im_ent_embs[lhs_idx]                           # (n_samples, d)
        assert torch.isclose(torch.sum(lhs_probs), torch.tensor(1.0))

        # p ~ \Pr(P | S=s) = (\sum_{O} \phi(s,P,O)) / (\sum_{P',O} \phi(s,P',O))
        re_lhs_unsq, im_lhs_unsq = re_lhs.unsqueeze(dim=1), im_lhs.unsqueeze(dim=1)
        p0_rhs = re_lhs_unsq * re_rel_embs                                      # (n_samples, n_r, d)
        p1_rhs = im_lhs_unsq * re_rel_embs                                      # (n_samples, n_r, d)
        p2_rhs = re_lhs_unsq * im_rel_embs                                      # (n_samples, n_r, d)
        p3_rhs = im_lhs_unsq * im_rel_embs                                      # (n_samples, n_r, d)
        u0 = torch.bmm(p0_rhs, re_re_ent.expand(n_samples, -1, -1))             # (n_samples, n_r, d)
        u1 = torch.bmm(p1_rhs, im_im_ent.expand(n_samples, -1, -1))             # (n_samples, n_r, d)
        u2 = torch.bmm(p2_rhs, im_im_ent.expand(n_samples, -1, -1))             # (n_samples, n_r, d)
        u3 = torch.bmm(p3_rhs, re_re_ent.expand(n_samples, -1, -1))             # (n_samples, n_r, d)
        u4 = torch.bmm((p0_rhs - p3_rhs), re_im_ent.expand(n_samples, -1, -1))  # (n_samples, n_r, d)
        v0 = torch.sum(p0_rhs * u0, dim=2)                                      # (n_samples, n_r)
        v1 = torch.sum(p1_rhs * u1, dim=2)                                      # (n_samples, n_r)
        v2 = torch.sum(p2_rhs * u2, dim=2)                                      # (n_samples, n_r)
        v3 = torch.sum(p3_rhs * u3, dim=2)                                      # (n_samples, n_r)
        v4 = torch.sum(p3_rhs * u0, dim=2)                                      # (n_samples, n_r)
        v5 = torch.sum(p2_rhs * u1, dim=2)                                      # (n_samples, n_r)
        v6 = torch.sum((p1_rhs + p2_rhs) * u4, dim=2)                           # (n_samples, n_r)
        rel_scores = v0 + v1 + v2 + v3 + 2.0 * (v6 + v5 - v4)
        rel_scores = rel_scores + self.e_eps
        rel_probs = rel_scores * torch.reciprocal(torch.sum(rel_scores, dim=1, keepdim=True))  # (n_samples,n_r)
        rel_idx = torch.multinomial(rel_probs, 1).squeeze(dim=1)                               # (n_samples)
        re_rel, im_rel = re_rel_embs[rel_idx],  im_rel_embs[rel_idx]                           # (n_samples, d)
        assert torch.allclose(torch.sum(rel_probs, dim=1), torch.tensor(1.0))

        # o ~ \Pr(O | S=s,P=p) = \phi(s,p,O) / (\sum_{O'} \phi(s,p,O'))
        p0_rhs = re_lhs * re_rel  # (n_samples, d)
        p1_rhs = im_lhs * re_rel  # (n_samples, d)
        p2_rhs = re_lhs * im_rel  # (n_samples, d)
        p3_rhs = im_lhs * im_rel  # (n_samples, d)
        rhs_scores = torch.square(
            (p0_rhs - p3_rhs) @ re_ent_embs.t() + (p1_rhs + p2_rhs) @ im_ent_embs.t()
        )  # (n_samples, n_e)
        rhs_scores = rhs_scores + self.eps  # (n_samples, n_e)
        rhs_probs = rhs_scores * torch.reciprocal(torch.sum(rhs_scores, dim=1, keepdim=True))  # (n_samples, n_e)
        rhs_idx = torch.multinomial(rhs_probs, 1).squeeze(dim=1)                               # (n_samples)
        assert torch.allclose(torch.sum(rhs_probs, dim=1), torch.tensor(1.0))

        return torch.stack([lhs_idx, rel_idx, rhs_idx], dim=1)


class TypedSquaredKBCModel(SquaredKBCModel):
    def typed_subject_partition_function(self, type_range: Tuple[int, int]) -> torch.Tensor:
        """
        Compute the partition function of the subject entities input distribution.

        :param type_range: The type range of the entities to select.
        :return: The partition function for subjects given the type range.
        """
        pass

    def typed_object_partition_function(self, type_range: Tuple[int, int]) -> torch.Tensor:
        """
        Compute the partition function of the object entities input distribution.

        :param type_range: The type range of the entities to select.
        :return: The partition function for objects given the type range.
        """
        pass

    def typed_relation_partition_function(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the partition function of the relations input distribution.

        :param mask: The mask of the relation types to select.
        :return: The partition function of the selected relation types..
        """
        pass

    def set_type_constraint_info(
        self,
        pred_to_domains: Dict[int, int],
        dom_to_types: Dict[int, Tuple[int, int]],
        dom_to_preds: Dict[int, List[int]],
        type_entity_ids: Dict[int, Tuple[int, int]],
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Set information about type constraints.
        """
        n_consistent_triples = 0
        for d_id in dom_to_types.keys():
            s_id, o_id = dom_to_types[d_id]
            s_range, o_range = type_entity_ids[s_id], type_entity_ids[o_id]
            d_rels = dom_to_preds[d_id]
            n_consistent_triples += (s_range[1] - s_range[0]) * len(d_rels) * (o_range[1] - o_range[0])
        self.eps = 1.0 / n_consistent_triples
        del self.e_eps
        del self.r_eps
        pred_to_domains_t = torch.tensor(
            [pred_to_domains[r] for r in sorted(pred_to_domains.keys())],
            dtype=torch.int64, device=device
        )
        dom_to_ent_types_t = torch.tensor(
            [dom_to_types[d] for d in sorted(dom_to_types.keys())],
            dtype=torch.int64, device=device
        )
        dom_to_pred_masks = dict()
        n_predicates = len(pred_to_domains)
        for d, preds, in dom_to_preds.items():
            mask = torch.zeros(n_predicates, dtype=torch.bool)
            mask[preds] = True
            dom_to_pred_masks[d] = mask
        dom_to_pred_masks_t = torch.stack(
            [dom_to_pred_masks[d] for d in sorted(dom_to_pred_masks.keys())]
        ).to(device)
        type_entity_ids_t = torch.tensor(
            [type_entity_ids[t] for t in sorted(type_entity_ids.keys())],
            dtype=torch.int64, device=device
        )
        self.register_buffer('pred_to_domains', pred_to_domains_t)
        self.register_buffer('dom_to_ent_types', dom_to_ent_types_t)
        self.register_buffer('dom_to_pred_masks', dom_to_pred_masks_t)
        self.register_buffer('type_to_ent_ranges', type_entity_ids_t)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        for bufname in ['pred_to_domains', 'dom_to_ent_types', 'dom_to_pred_masks', 'type_to_ent_ranges']:
            self.register_buffer(bufname, torch.empty_like(state_dict[bufname]))
        super().load_state_dict(state_dict, strict=strict)

    def get_queries(self, x: torch.Tensor, target: str = 'rhs') -> Tuple[torch.Tensor, torch.Tensor]:
        # Evaluate the circuit partially based on the target flag
        lhs, rel, rhs = self.index_embeddings(x)
        qs = self.eval_circuit_partial(lhs, rel, rhs, target=target)
        d = self.pred_to_domains[x[:, 1]]
        so_type_ids = self.dom_to_ent_types[d]
        s_type_id, o_type_id = so_type_ids[:, 0], so_type_ids[:, 1]
        if target == 'rhs':
            target_domain = self.type_to_ent_ranges[o_type_id]
        elif target == 'rel':
            target_domain = self.dom_to_pred_masks[d]
        elif target == 'lhs':
            target_domain = self.type_to_ent_ranges[s_type_id]
        else:
            raise ValueError("Invalid target value")
        return qs, target_domain

    def get_candidates(
            self,
            chunk_begin: int,
            chunk_size: int,
            target: str = 'rhs',
            indices: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cs = super().get_candidates(chunk_begin, chunk_size, target=target, indices=indices, device=device)
        if indices is not None:
            return cs, None
        idx = torch.arange(chunk_begin, chunk_begin + chunk_size, device=device)
        return cs, idx

    def eval_circuit_all(
            self,
            qs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            cs: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]
    ) -> torch.Tensor:
        if not isinstance(qs, tuple):
            return super().eval_circuit_all(qs, cs)
        scores = super().eval_circuit_all(qs[0], cs[0])  # (b, n)
        candidate_indices = cs[1].unsqueeze(dim=0)  # (1, n)
        target_domain = qs[1].unsqueeze(dim=1)  # (b, 1, 2) [int] for entities and (b, 1, n) [bool] for predicates
        if target_domain.dtype == torch.bool:
            raise NotImplementedError("Evaluation over all predicates not implemented yet")
        else:
            inconsistent_mask = (candidate_indices < target_domain[..., 0]) | (candidate_indices >= target_domain[..., 1])
        scores[inconsistent_mask] = 0.0
        return scores

    def log_likelihood(
            self,
            x: torch.Tensor,
            con_rhs: bool = False,
            con_rel: bool = False,
            con_lhs: bool = False,
            return_ll: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Get the embeddings of the input triples
        lhs, rel, rhs = self.index_embeddings(x)

        # Evaluate the circuit
        scores = self.eval_circuit(lhs, rel, rhs)
        log_scores = torch.log(scores + self.eps)

        # Compute the partition functions of the input functionals
        if con_rhs or return_ll:
            z_rhs_ent = list()
            for i in range(len(self.type_to_ent_ranges)):
                z_rhs_ent.append(self.typed_object_partition_function(self.type_to_ent_ranges[i]))
            z_rhs_ent = torch.stack(z_rhs_ent)  # (n-types, ...)
        else:
            z_rhs_ent = None
        if con_rel or return_ll:
            z_rel = list()
            for d in range(len(self.dom_to_pred_masks)):
                z_rel.append(self.typed_relation_partition_function(self.dom_to_pred_masks[d]))
            z_rel = torch.stack(z_rel)  # (n-doms, ...)
        else:
            z_rel = None
        if con_lhs or return_ll:
            z_lhs_ent = list()
            for i in range(len(self.type_to_ent_ranges)):
                z_lhs_ent.append(self.typed_subject_partition_function(self.type_to_ent_ranges[i]))
            z_lhs_ent = torch.stack(z_lhs_ent)  # (n-types, ...)
        else:
            z_lhs_ent = None

        # Compute the partition function, if needed
        log_probs = None
        overall_z = 0.0
        if return_ll:
            for d in range(len(self.dom_to_ent_types)):
                s_type_id, o_type_id = self.dom_to_ent_types[d]
                m_z_lhs_ent, m_z_rel, m_z_rhs_ent = z_lhs_ent[s_type_id], z_rel[d], z_rhs_ent[o_type_id]
                overall_z += self.eval_circuit_partition_function(m_z_lhs_ent, m_z_rel, m_z_rhs_ent)
            log_probs = log_scores - torch.log1p(overall_z)

        if not con_rhs and not con_rel and not con_lhs:
            return log_probs, [None, None, None]

        # Get unique domain ids of triples in the given batch
        log_con_probs = [list(), list(), list()]
        domains = self.pred_to_domains[x[:, 1].detach()]
        batch_domains = torch.unique(domains)
        batch_indices = list()
        # Compute conditional probabilities, if needed
        for d in batch_domains:
            mask = domains == d
            batch_indices.append(torch.where(mask)[0])
            m_lhs = tuple([x[mask] for x in lhs]) if isinstance(lhs, Tuple) else lhs[mask]
            m_rel = tuple([x[mask] for x in rel]) if isinstance(rel, Tuple) else rel[mask]
            m_rhs = tuple([x[mask] for x in rhs]) if isinstance(rhs, Tuple) else rhs[mask]
            m_log_scores = log_scores[mask]
            s_type_id, o_type_id = self.dom_to_ent_types[d]
            # Perform CON inference
            if con_rhs:
                e_eps = self.eps * (self.type_to_ent_ranges[o_type_id, 1] - self.type_to_ent_ranges[o_type_id, 0])
                m_z_rhs_ent = z_rhs_ent[o_type_id]
                mar_rhs = self.marginalized_score(m_lhs, m_rel, m_rhs, None, None, m_z_rhs_ent, target='rhs')
                log_con_probs[0].append(m_log_scores - torch.log(mar_rhs + e_eps))
            if con_rel:
                r_mask = self.dom_to_pred_masks[d]
                n_rels = torch.count_nonzero(r_mask)
                if n_rels == 1:
                    log_con_probs[1].append(torch.zeros_like(m_log_scores))
                else:
                    r_eps = self.eps * n_rels
                    m_z_rel = z_rel[d]
                    mar_rel = self.marginalized_score(m_lhs, m_rel, m_rhs, None, m_z_rel, None, target='rel')
                    log_con_probs[1].append(m_log_scores - torch.log(mar_rel + r_eps))
            if con_lhs:
                e_eps = self.eps * (self.type_to_ent_ranges[s_type_id, 1] - self.type_to_ent_ranges[s_type_id, 0])
                m_z_lhs_ent = z_lhs_ent[s_type_id]
                mar_lhs = self.marginalized_score(m_lhs, m_rel, m_rhs, m_z_lhs_ent, None, None, target='lhs')
                log_con_probs[2].append(m_log_scores - torch.log(mar_lhs + e_eps))

        batch_indices = torch.concat(batch_indices)
        reorder_perm = torch.empty_like(batch_indices)
        reorder_perm[batch_indices] = torch.arange(len(batch_indices), device=batch_indices.device)
        log_con_probs[0] = torch.concat(log_con_probs[0])[reorder_perm] if log_con_probs[0] else None
        log_con_probs[1] = torch.concat(log_con_probs[1])[reorder_perm] if log_con_probs[1] else None
        log_con_probs[2] = torch.concat(log_con_probs[2])[reorder_perm] if log_con_probs[2] else None

        return log_probs, log_con_probs


class TypedSquaredCP(SquaredCP, TypedSquaredKBCModel):
    def typed_subject_partition_function(self, type_range: Tuple[int, int]) -> torch.Tensor:
        lhs_ent = self.ent_embeddings.weight[type_range[0]:type_range[1], :self.rank]
        z_lhs_ent = torch.mm(lhs_ent.t(), lhs_ent)
        return z_lhs_ent

    def typed_object_partition_function(self, type_range: Tuple[int, int]) -> torch.Tensor:
        rhs_ent = self.ent_embeddings.weight[type_range[0]:type_range[1], self.rank:]
        z_rhs_ent = torch.mm(rhs_ent.t(), rhs_ent)
        return z_rhs_ent

    def typed_relation_partition_function(self, mask: torch.Tensor) -> torch.Tensor:
        rel = self.rel_embeddings.weight[mask]
        z_rel = torch.mm(rel.t(), rel)
        return z_rel


class TypedSquaredComplEx(SquaredComplEx, TypedSquaredKBCModel):
    def typed_subject_partition_function(self, type_range: Tuple[int, int]) -> torch.Tensor:
        weight = self.ent_embeddings.weight[type_range[0]:type_range[1]]
        re_ent = weight[:, :self.ri_rank]
        im_ent = weight[:, self.ri_rank:]
        re_re_ent = torch.mm(re_ent.t(), re_ent)  # (d, d)
        re_im_ent = torch.mm(re_ent.t(), im_ent)  # (d, d)
        im_im_ent = torch.mm(im_ent.t(), im_ent)  # (d, d)
        z_ent = torch.stack([re_re_ent, re_im_ent, im_im_ent])
        return z_ent

    def typed_object_partition_function(self, type_range: Tuple[int, int]) -> torch.Tensor:
        return self.typed_subject_partition_function(type_range)

    def typed_relation_partition_function(self, mask: torch.Tensor) -> torch.Tensor:
        weight = self.rel_embeddings.weight[mask]
        re_rel = weight[:, :self.ri_rank]
        im_rel = weight[:, self.ri_rank:]
        re_re_rel = torch.mm(re_rel.t(), re_rel)  # (d, d)
        re_im_rel = torch.mm(re_rel.t(), im_rel)  # (d, d)
        im_im_rel = torch.mm(im_rel.t(), im_rel)  # (d, d)
        z_rel = torch.stack([re_re_rel, re_im_rel, im_im_rel])
        return z_rel

    def eval_circuit_partition_function(self, z_lhs: torch.Tensor, z_rel: torch.Tensor, z_rhs: torch.Tensor) -> torch.Tensor:
        (re_re_lhs_ent, re_im_lhs_ent, im_im_lhs_ent) = z_lhs
        (re_re_rhs_ent, re_im_rhs_ent, im_im_rhs_ent) = z_rhs
        (re_re_rel, re_im_rel, im_im_rel) = z_rel
        im_re_lhs_ent = re_im_lhs_ent.t()
        im_re_rhs_ent = re_im_rhs_ent.t()
        #
        x0 = re_re_lhs_ent * re_re_rel
        x1 = im_im_lhs_ent * re_re_rel
        x2 = re_re_lhs_ent * im_im_rel
        x3 = im_im_lhs_ent * im_im_rel
        sq = (x0 + x3) * re_re_rhs_ent + (x1 + x2) * im_im_rhs_ent
        #
        v1 = re_re_rel * re_im_rhs_ent
        v2 = im_im_rel * im_re_rhs_ent
        v3 = re_im_rhs_ent * re_re_lhs_ent
        v4 = re_re_rhs_ent * re_im_lhs_ent
        v5 = im_im_rhs_ent * im_re_lhs_ent
        v6 = im_re_rhs_ent * im_im_lhs_ent
        co = (v1 - v2) * re_im_lhs_ent + (v3 - v4 + v5 - v6) * re_im_rel
        z = sq + 2.0 * co
        return torch.sum(z.view(-1, self.ri_rank * self.ri_rank), dim=1, keepdim=True)
