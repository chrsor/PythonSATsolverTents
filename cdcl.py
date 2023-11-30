#!/usr/bin/env python3
import argparse
import itertools
import os
import signal
from collections import deque, Counter
import numpy as np
import time
import sys

from preprocessing import Preprocessor


class Clause:
    def __init__(self, lits, wl_1=None, wl_2=None, is_learned=False, lit_blocks_dist=0):
        self.lits = lits
        self.length = len(self.lits)
        self.wl_1, self.wl_2 = wl_1, wl_2
        self.is_learned = is_learned
        self.lit_blocks_dist = lit_blocks_dist

        if not wl_1 and not wl_2:  # if (not wl_1) and (not wl_2)
            if self.length > 1:
                self.wl_1 = 0
                self.wl_2 = 1
            elif self.length:
                self.wl_1 = self.wl_2 = 0

    def update_wl(self, assignment, var):
        if var == abs(self.lits[self.wl_2]):
            temp = self.wl_1
            self.wl_1 = self.wl_2
            self.wl_2 = temp

        if self.lits[self.wl_1] == assignment[abs(self.lits[self.wl_1])] or self.lits[self.wl_2] == assignment[
            abs(self.lits[self.wl_2])]:
            return True, self.lits[self.wl_1], False

        if -self.lits[self.wl_1] == assignment[abs(self.lits[self.wl_1])] and -self.lits[self.wl_2] == assignment[
            abs(self.lits[self.wl_2])]:
            return False, self.lits[self.wl_1], False

        if -self.lits[self.wl_1] == assignment[abs(self.lits[self.wl_1])] and assignment[
            abs(self.lits[self.wl_2])] == 0:
            prev_wl_1 = self.wl_1
            for wl in [(self.wl_1 + x) % self.length for x in range(self.length)]:
                if -self.lits[wl] == assignment[abs(self.lits[wl])] or wl == self.wl_2:
                    continue
                self.wl_1 = wl
                break

            if prev_wl_1 == self.wl_1:
                return True, self.lits[self.wl_1], True

            return True, self.lits[self.wl_1], False


class Formula:
    def __init__(self, cnf):
        self.cnf = cnf  # [[literal_x, literal_y,...,literal_z],[literal_a,literal_b],...]. List of lists.
        self.clauses_as_objects = [Clause(lits) for lits in self.cnf]
        self.learned_clauses = []
        self.conflict_clauses = []
        self.deleted_clauses = []
        self.vars = set()  # Set of variables, which are in the current formula.
        self.watched_lits = {}  # {literal_x : [clause_1, clause_2...], ... ]
        self.queue_of_unit_clauses = deque()
        self.assignment_history = deque()
        self.assignment, self.previous, self.level, self.neg_lits, self.pos_lits = None, None, None, None, None
        self.s = Counter()

        for c in self.clauses_as_objects:
            if c.wl_1 == c.wl_2:
                self.queue_of_unit_clauses.append((c, c.lits[c.wl_2]))

            for l in c.lits:
                var = abs(l)
                self.vars.add(var)

                if var not in self.watched_lits:
                    self.watched_lits[var] = []

                if c.lits[c.wl_1] == l or c.lits[c.wl_2] == l:
                    if c not in self.watched_lits[var]:
                        self.watched_lits[var].append(c)

        end_var = max(self.vars)
        self.assignment = [0] * (end_var + 1)
        self.previous = [None] * (end_var + 1)
        self.level = [-1] * (end_var + 1)
        self.neg_lits = np.zeros((end_var + 1), dtype=np.float64)
        self.pos_lits = np.zeros((end_var + 1), dtype=np.float64)

    def everything_assigned(self):
        return len(self.vars) == len(self.assignment_history)

    def unit_prop(self, level):
        prop_lits = []
        while self.queue_of_unit_clauses:
            unit_clause, free_unit_clause_literal = self.queue_of_unit_clauses.popleft()
            prop_lits.append(free_unit_clause_literal)
            self.previous[abs(free_unit_clause_literal)] = unit_clause

            res, conflict = self.assign(free_unit_clause_literal, level)
            if not res:
                return prop_lits, conflict

        return prop_lits, None

    def assign(self, lit, level):
        self.assignment_history.append(lit)
        self.assignment[abs(lit)] = lit
        self.level[abs(lit)] = level

        watched = self.watched_lits[abs(lit)][:]

        for c in watched:
            not_unsat, wl, is_unit = c.update_wl(self.assignment, abs(lit))

            if not_unsat:
                if abs(wl) != abs(lit):
                    if c not in self.watched_lits[abs(wl)]:
                        self.watched_lits[abs(wl)].append(c)

                    self.watched_lits[abs(lit)].remove(c)

                if is_unit:
                    if c.lits[c.wl_2] not in [x[1] for x in self.queue_of_unit_clauses]:
                        self.queue_of_unit_clauses.append((c, c.lits[c.wl_2]))

            if not not_unsat:
                return False, c

        return True, None

    def pick_lit(self, heuristic):
        if heuristic == 'VSIDS':
            return self.vsids()

        if heuristic == 'VMTF':
            if not self.s:
                self.s = Counter()
                for clause in self.cnf:
                    abs_clause = map(abs, clause)
                    t = Counter(abs_clause)
                    self.s.update(t)

                self.priority_queue = sorted(self.s, key=self.s.get)
            return list(filter(lambda v: self.assignment[v] == 0, self.priority_queue))[0]

    def vsids(self):
        picked_lit = None
        c = 0
        for var in self.vars:
            if self.assignment[var] == 0:
                if self.pos_lits[var] > c:
                    picked_lit = var
                    c = self.pos_lits[var]

                if self.neg_lits[var] >= c:
                    picked_lit = -var
                    c = self.neg_lits[var]

        return picked_lit

    def update_vmtf(self, learned):
        for lit in learned:
            self.s[abs(lit)] += 1

        learned_vars = list(map(abs, learned))
        learned_vars.sort(key=self.s.get)
        n = len(learned_vars) if len(learned_vars) <= 8 else 8
        vars_to_move = learned_vars[-n:]

        for var in vars_to_move:
            self.priority_queue.append(self.priority_queue.pop(self.priority_queue.index(var)))

    def restart(self):
        self.queue_of_unit_clauses.clear()
        self.backtrack(level=0)

    def backtrack(self, level):
        while self.assignment_history and self.level[abs(self.assignment_history[-1])] > level:
            lit = self.assignment_history.pop()
            self.assignment[abs(lit)] = 0
            self.previous[abs(lit)] = None
            self.level[abs(lit)] = -1

    def delete_unnecessary_clauses(self, lit_blocks_dist_limit):
        lit_blocks_dist_limit = int(lit_blocks_dist_limit)
        learned_clauses = []
        deleted_clauses = []
        for c in self.learned_clauses:
            if c.lit_blocks_dist > lit_blocks_dist_limit:
                self.watched_lits[abs(c.lits[c.wl_1])].remove(c)
                if c.wl_1 != c.wl_2:
                    self.watched_lits[abs(c.lits[c.wl_2])].remove(c)

            else:
                learned_clauses.append(c)
                deleted_clauses.append(c)
        self.learned_clauses = learned_clauses
        self.deleted_clauses = deleted_clauses

    @staticmethod
    def resolve(c1, c2, l):
        clause1 = set(c1)
        clause2 = set(c2)
        clause1.remove(-l)
        clause2.remove(l)
        return list(clause1.union(clause2))

    def deal_with_conflict(self, heuristic, conflict, level):
        if level == 0:
            return -1

        assert_clause_lits = conflict.lits
        curr_assgnmt = deque(self.assignment_history)
        while len([lit for lit in assert_clause_lits if self.level[abs(lit)] == level]) > 1:
            while True:
                lit = curr_assgnmt.pop()
                if -lit in assert_clause_lits:
                    assert_clause_lits = self.resolve(assert_clause_lits, self.previous[abs(lit)].lits, lit)
                    break

        assert_level = 0
        unit_lit = None
        wl_2 = None
        curr_level = [False] * (level + 1)
        for i, l in enumerate(assert_clause_lits):
            if assert_level < self.level[abs(l)] < level:
                assert_level = self.level[abs(l)]

            if self.level[abs(l)] == level:
                unit_lit = l
                wl_2 = i

            if not curr_level[self.level[abs(l)]]:
                curr_level[self.level[abs(l)]] = True

            if heuristic == 'VSIDS':

                self.pos_lits = self.pos_lits * 0.9
                self.neg_lits = self.neg_lits * 0.9
                if l > 0:
                    self.pos_lits[l] += 1

                else:
                    self.neg_lits[abs(l)] += 1

        lit_blocks_dist = sum(curr_level)

        wl_1 = None
        if len(assert_clause_lits) > 1:
            curr_assgnmt = deque(self.assignment_history)
            res = False
            while curr_assgnmt:
                l = curr_assgnmt.pop()
                if self.level[abs(l)] == assert_level:
                    for i, cl in enumerate(assert_clause_lits):
                        if abs(l) == abs(cl):
                            wl_1 = i
                            res = True
                            break

                if res:
                    break

        else:
            wl_1 = wl_2

        assert_clause = Clause(assert_clause_lits, wl_1=wl_1, wl_2=wl_2, is_learned=True,
                               lit_blocks_dist=lit_blocks_dist)
        self.watched_lits[abs(assert_clause.lits[assert_clause.wl_1])].append(assert_clause)
        if assert_clause.wl_1 != assert_clause.wl_2:
            self.watched_lits[abs(assert_clause.lits[assert_clause.wl_2])].append(assert_clause)

        self.conflict_clauses.append(assert_clause)
        self.learned_clauses.append(assert_clause)
        if heuristic == 'VMTF':
            self.update_vmtf(assert_clause.lits)
        self.queue_of_unit_clauses.clear()
        self.queue_of_unit_clauses.append((assert_clause, unit_lit))

        return assert_level


def read_file(filename, preprocessing=False):
    if preprocessing:
        p = Preprocessor(filename)
        return p.preprocess()
    else:
        with open(filename) as f:
            lines = [
                line.strip().split() for line in f.readlines()
                if (not (line.startswith('c') or line.startswith('0') or line.startswith('p')) and line != '\n')
            ]

        clauses = []

        doc_as_one_line = [i for sublist in lines for i in sublist]
        l_g = [list(y) for x, y in itertools.groupby(doc_as_one_line, lambda z: z == '0') if not x]

        for l in l_g:
            clause = list(map(int, l))
            clauses.append(clause)
        return clauses


def cdcl(formula, heuristic, restart_after, lit_blocks_dist_limit):
    level, num_of_decisions, num_of_unit_props, num_of_restarts, num_of_conflicts = 0, 0, 0, 0, 0
    prop_lits, conflict = formula.unit_prop(level)
    num_of_unit_props += len(prop_lits)

    if conflict:
        return False, [], num_of_decisions, num_of_unit_props, num_of_restarts

    while not formula.everything_assigned():
        picked_lit = formula.pick_lit(heuristic)
        level += 1

        formula.assign(picked_lit, level)
        num_of_decisions += 1

        prop_lits, conflict = formula.unit_prop(level)
        num_of_unit_props += len(prop_lits)

        while conflict:
            num_of_conflicts += 1

            if num_of_conflicts == restart_after:
                num_of_conflicts = 0
                restart_after = int(restart_after * 1.15)
                num_of_restarts += 1
                level = 0
                formula.restart()
                formula.delete_unnecessary_clauses(lit_blocks_dist_limit)
                break

            level_of_backtracking = formula.deal_with_conflict(heuristic, conflict, level)
            if level_of_backtracking < 0:
                return False, [], num_of_decisions, num_of_unit_props, num_of_restarts

            formula.backtrack(level_of_backtracking)
            level = level_of_backtracking

            prop_lits, conflict = formula.unit_prop(level)
            num_of_unit_props += len(prop_lits)

    return True, list(formula.assignment_history), num_of_decisions, num_of_unit_props, num_of_restarts


def solve(filename, heuristic, restart_after, lit_blocks_dist_limit, proof=False, benchmark=False, preprocessing=False):
    formula = read_file(filename, preprocessing=preprocessing)
    cnf = Formula(formula)
    if benchmark:
        signal.alarm(60)
    start_time = time.time()
    res, model, num_of_decisions, num_of_unit_props, num_of_restarts = cdcl(cnf, heuristic, restart_after,
                                                                            lit_blocks_dist_limit)
    spent_time = time.time() - start_time

    if res:
        model.sort(key=abs)  # Optional
        print('Processing ', filename)
        print('-----------')
        print('SATISFIABLE')
        print('-----------')
        print('Solution: ', model)
    else:
        if proof:
            i = input("Do you like to get a Proof for Unsat? Then write y: ")

        # DRUP-Proof
        if (i == 'y'):
            fp = open("proof.txt", "w")
            set_l = set(cnf.learned_clauses)
            set_c = set(cnf.conflict_clauses)
            for l in range(len(cnf.learned_clauses)):
                t = set(cnf.learned_clauses[l]) & set(cnf.conflict_clauses)
                if (t): # Then it is a conflict clause => RUP-Format
                    for x in range(len(x[z] for x in cnf.learned_clauses)):
                        z = 0
                        fp.write("" + x[z])
                        z += z
                    fp.write("" + "0\n")
                else: # Then it is a deleted clause => 'd' + clause
                    fp.write("d")
                    for x in range(len(x[z] for x in cnf.learned_clauses)):
                        b = 0
                        fp.write("" + x[b])
                        b += b
                    fp.write("" + "0\n")
            fp.write("0")
            fp.close()
        print('-----------')
        print('UNSATISFIABLE')
        print('-----------')


    print('Time: ', spent_time, ' secs')
    print('Num_of_decicions: ', num_of_decisions)
    print('Num_of_propagations: ', num_of_unit_props)
    print('Num_of_restarts: ', num_of_restarts)

    return res, model, spent_time, num_of_decisions, num_of_unit_props, num_of_restarts


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def main():
    p = argparse.ArgumentParser()
    p.add_argument('filename')
    p.add_argument('-hc', '--heuristic', default='VSIDS')
    p.add_argument('-r', '--restart_after', default=100)
    p.add_argument('-lbd', '--lit_blocks_dist_limit', default=3)
    p.add_argument('-bd', '--benchmark_dir', default=None)
    p.add_argument('-pre', '--preprocessing', default=False)
    p.add_argument('-prf', '--proof', default=False)

    args = p.parse_args()

    if args.benchmark_dir is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        results = []
        for subdir, dirs, files in os.walk(args.benchmark_dir):
            for file in files:
                filename = os.path.join(subdir, file)
                print("START: {}".format(filename))
                try:
                    res, model, spent_time, num_of_decisions, num_of_unit_props, num_of_restarts = solve(
                        filename,
                        args.heuristic,
                        args.restart_after,
                        args.lit_blocks_dist_limit,
                        proof=args.proof,
                        benchmark=True,
                        preprocessing=args.preprocessing,
                    )
                    results.append([filename, res, model, spent_time, num_of_decisions, num_of_unit_props, num_of_restarts])
                except TimeoutException:
                    results.append([filename, None, None, None, None, None, None])
                    print("{}: TIMEOUT".format(filename))
                    continue
                else:
                    signal.alarm(0)

        results = np.array(results, dtype=object).reshape(-1, 7)
        np.save('benchmarks/{}-{}-{}.npy'.format(args.heuristic, args.restart_after, args.preprocessing), results)
    else:
        solve(args.filename, args.heuristic, args.restart_after, args.lit_blocks_dist_limit)


main()
