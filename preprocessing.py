import itertools


class Preprocessor:

    def __init__(self, path):
        self.clauses = self.initialize(path)


    def initialize(self, path):
        clauses = []
        with open(path) as f:
            for line in f.readlines():
                if (not (line.startswith('c') or
                         line.strip().startswith('0') or
                         line.startswith('p')) and
                        line != '\n'):
                    clause = [int(literal) for literal in line.strip().split() if int(literal) != 0]
                    clauses.append(set(clause))

        return clauses


    def preprocess(self, variable_elimination=False):
        def subsume_pair(pair, clause_list, subsumed_indices):
            longer_clause = pair[0] if pair[0] >= pair[1] else pair[1]
            shorter_clause = pair[0] if pair[0] < pair[1] else pair[1]
            if clause_list[shorter_clause] != clause_list[longer_clause] and shorter_clause not in subsumed_indices and longer_clause not in subsumed_indices:
                for literal in clause_list[longer_clause]:
                    if (literal * - 1) in clause_list[shorter_clause]:
                        shorter_clause_copy = clause_list[shorter_clause].copy()
                        shorter_clause_copy.remove(literal * - 1)
                        if shorter_clause_copy.issubset(clause_list[longer_clause]):
                            longer_clause_copy = clause_list[longer_clause].copy()
                            longer_clause_copy.remove(literal)
                            return longer_clause_copy

            return set()

        made_progress = True
        new_clause_list = []
        while made_progress:
            made_progress = False
            clause_list = new_clause_list if len(new_clause_list) > 0 else self.clauses
            new_clause_list = []
            not_subsumed_clause_list = []
            combinations = list(itertools.combinations(range(len(clause_list)), 2))
            combinations_copy = combinations[:]
            subsumed_indices = []
            subsumed_clause_list = []

            for clause_pair in combinations:
                has_subsumed = subsume_pair(clause_pair, clause_list, subsumed_indices)
                if has_subsumed != set():
                    subsumed_indices.append(clause_pair[0])
                    subsumed_indices.append(clause_pair[1])
                    subsumed_clause_list.append(has_subsumed)
                    made_progress = True
                    combinations_copy.remove(clause_pair)

            for clause_pair in combinations_copy:
                if clause_pair[0] not in subsumed_indices and \
                        clause_list[clause_pair[0]] not in not_subsumed_clause_list:
                    not_subsumed_clause_list.append(clause_list[clause_pair[0]])
                    subsumed_indices.append(clause_pair[0])
                if clause_pair[1] not in subsumed_indices and \
                        clause_list[clause_pair[1]] not in not_subsumed_clause_list:
                    not_subsumed_clause_list.append(clause_list[clause_pair[1]])
                    subsumed_indices.append(clause_pair[1])

            new_clause_list.extend(not_subsumed_clause_list)
            new_clause_list.extend(subsumed_clause_list)

            if not new_clause_list:
                new_clause_list = clause_list

        print("Removed {} clauses".format(abs(len(new_clause_list) - len(self.clauses))))
        return [list(clause) for clause in new_clause_list]
