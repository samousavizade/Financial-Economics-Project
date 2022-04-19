import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from risk_estimators import RiskEstimators
# from mlfinlab.portfolio_optimization.estimators.risk_estimators import RiskEstimators


class TheoryImpliedCorrelation:
    def __init__(self):
        return

    def tic_correlation(self, tree_struct, corr_matrix, tn_relation, kde_bwidth=0.01):
        lnkage_object = self._get_linkage_corr(tree_struct, corr_matrix)

        ti_correlation = self._link2corr(lnkage_object, corr_matrix.index)

        risk_estim = RiskEstimators()

        ti_correlation_denoised = risk_estim.denoise_covariance(ti_correlation, tn_relation=tn_relation,
                                                                kde_bwidth=kde_bwidth)

        return ti_correlation_denoised

    @staticmethod
    def corr_dist(corr0, corr1):
        prod_trace = np.trace(np.dot(corr0, corr1))
        frob_product = np.linalg.norm(corr0, ord='fro')
        frob_product *= np.linalg.norm(corr1, ord='fro')

        distance = 1 - prod_trace / frob_product

        return distance

    def _get_linkage_corr(self, tree_struct, corr_matrix):
        if len(np.unique(tree_struct.iloc[:, -1])) > 1:
            tree_struct = tree_struct.copy(deep=True)
            tree_struct['All'] = 0

        global_linkage = np.empty(shape=(0, 4))
        tree_levels = [[tree_struct.columns[i-1], tree_struct.columns[i]] for i in range(1, tree_struct.shape[1])]
        distance_matrix = ((1 - corr_matrix) / 2)**(1/2)
        global_elements = distance_matrix.index.tolist()

        for level in tree_levels:
            grouped_level = tree_struct[level].drop_duplicates(level[0]).set_index(level[0]).groupby(level[1])

            for high_element, grouped_elements in grouped_level:
                grouped_elements = grouped_elements.index.tolist()
                if len(grouped_elements) == 1:
                    global_elements[global_elements.index(grouped_elements[0])] = high_element

                    distance_matrix = distance_matrix.rename({grouped_elements[0]: high_element}, axis=0)
                    distance_matrix = distance_matrix.rename({grouped_elements[0]: high_element}, axis=1)

                    continue

                local_distance = distance_matrix.loc[grouped_elements, grouped_elements]

                distance_vec = ssd.squareform(local_distance, force='tovector',
                                              checks=(not np.allclose(local_distance, local_distance.T)))

                local_linkage = sch.linkage(distance_vec, optimal_ordering=True)

                local_linkage_transformed = self._link_clusters(global_linkage, local_linkage, global_elements,
                                                                grouped_elements)

                global_linkage = np.append(global_linkage, local_linkage_transformed, axis=0)

                global_elements += range(len(global_elements), len(global_elements) + len(local_linkage_transformed))

                distance_matrix = self._update_dist(distance_matrix, global_linkage, local_linkage_transformed,
                                                    global_elements)

                global_elements[-1] = high_element

                distance_matrix.columns = distance_matrix.columns[:-1].tolist() + [high_element]
                distance_matrix.index = distance_matrix.columns

        global_linkage = np.array([*map(tuple, global_linkage)],
                                  dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])

        return global_linkage

    @staticmethod
    def _link_clusters(global_linkage, local_linkage, global_elements, grouped_elements):
        num_atoms = len(global_elements) - global_linkage.shape[0]

        local_linkage_tr = local_linkage.copy()

        for link in range(local_linkage_tr.shape[0]):
            atom_counter = 0

            for j in range(2):

                if local_linkage_tr[link, j] < len(grouped_elements): 
                    local_linkage_tr[link, j] = global_elements.index(grouped_elements[int(local_linkage_tr[link, j])])

                else:
                    local_linkage_tr[link, j] += -len(grouped_elements) + len(global_elements)

                if local_linkage_tr[link, j] < num_atoms: 
                    atom_counter += 1

                else:  
                    if local_linkage_tr[link, j] - num_atoms < global_linkage.shape[0]:
                        atom_counter += global_linkage[int(local_linkage_tr[link, j]) - num_atoms, 3]

                    else:
                        atom_counter += local_linkage_tr[int(local_linkage_tr[link, j]) - len(global_elements), 3]

            local_linkage_tr[link, 3] = atom_counter

        return local_linkage_tr

    @staticmethod
    def _update_dist(distance_matrix, global_linkage, local_linkage_tr, global_elements, criterion=None):
        num_atoms = len(global_elements) - global_linkage.shape[0]
        new_items = global_elements[-local_linkage_tr.shape[0]:]

        for i in range(local_linkage_tr.shape[0]):
            elem_1, elem_2 = global_elements[int(local_linkage_tr[i, 0])], global_elements[int(local_linkage_tr[i, 1])]

            if criterion is None:

                if local_linkage_tr[i, 0] < num_atoms: 
                    elem_1_weight = 1

                else: 
                    elem_1_weight = global_linkage[int(local_linkage_tr[i, 0]) - num_atoms, 3]

                if local_linkage_tr[i, 1] < num_atoms:
                    elem_2_weight = 1

                else:  
                    elem_2_weight = global_linkage[int(local_linkage_tr[i, 1]) - num_atoms, 3]

                dist_vector = (distance_matrix[elem_1] * elem_1_weight + distance_matrix[elem_2] * elem_2_weight) / \
                              (elem_1_weight + elem_2_weight)

            else:
                dist_vector = criterion(distance_matrix[[elem_1, elem_2]], axis=1)

            distance_matrix[new_items[i]] = dist_vector

            distance_matrix.loc[new_items[i]] = dist_vector

            distance_matrix.loc[new_items[i], new_items[i]] = 0

            distance_matrix = distance_matrix.drop([elem_1, elem_2], axis=0)
            distance_matrix = distance_matrix.drop([elem_1, elem_2], axis=1)

        return distance_matrix

    @staticmethod
    def _get_atoms(linkage, element):
        element_list = [element]

        while True:
            item_ = max(element_list)

            if item_ > linkage.shape[0]:
                element_list.remove(item_)

                element_list.append(linkage['i0'][item_ - linkage.shape[0] - 1])
                element_list.append(linkage['i1'][item_ - linkage.shape[0] - 1])

            else: 
                break

        return element_list

    def _link2corr(self, linkage, element_index):
        corr_matrix = pd.DataFrame(np.eye(linkage.shape[0]+1), index=element_index, columns=element_index, dtype=float)

        for link in range(linkage.shape[0]):
            el_x = self._get_atoms(linkage, linkage['i0'][link])
            el_y = self._get_atoms(linkage, linkage['i1'][link])
            corr_matrix.loc[element_index[el_x], element_index[el_y]] = 1 - 2 * linkage['dist'][link]**2
            corr_matrix.loc[element_index[el_y], element_index[el_x]] = 1 - 2 * linkage['dist'][link]**2

        return corr_matrix