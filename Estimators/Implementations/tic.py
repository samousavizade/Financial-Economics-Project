# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from mlfinlab.portfolio_optimization.estimators.risk_estimators import RiskEstimators


class TheoryImpliedCorrelation:
    """
    This class implements the Theory-Implied Correlation (TIC) algorithm and the correlation matrix distance
    introduced by Herdin and Bonek. It is reproduced with modification from the following paper:
    `Marcos Lopez de Prado “Estimation of Theory-Implied Correlation Matrices”, (2019).
    <https://papers.ssrn.com/abstract_id=3484152>`_.
    """

    def __init__(self):
        """
        Initialize
        """


        pass

    def tic_correlation(self, gics, corr, tn_relation, kde_bwidth=0.01):
        """
        Calculates the Theory-Implied Correlation (TIC) matrix.
        Includes three steps.
        In the first step, the theoretical tree graph structure of the assets is fit on the evidence
        presented by the empirical correlation matrix.
        The result of the first step is a binary tree (dendrogram) that sequentially clusters two elements
        together, while measuring how closely together the two elements are, until all elements are
        subsumed within the same cluster.
        In the second step, a correlation matrix is derived from the linkage object.
        Each cluster in the global linkage object is decomposed into two elements,
        which can be either atoms or other clusters. Then the off-diagonal correlation between two
        elements is calculated based on the distances between them.
        In the third step, the correlation matrix is de-noised.
        This is done by fitting the Marcenko-Pastur distribution to the eigenvalues of the matrix, calculating the
        maximum theoretical eigenvalue as a threshold and shrinking the eigenvalues higher than a set threshold.
        This algorithm is implemented in the RiskEstimators class.
        :param tree_struct: (pd.dataframe) The tree graph that represents the structure of the assets
        :param corr_matrix: (pd.dataframe) The empirical correlation matrix of the assets
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    correlation matrix
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE for de-noising the correlation matrix
                                   (0.01 by default)
        :return: (np.array) Theory-Implied Correlation matrix
        """

        lnk0=TheoryImpliedCorrelation()._get_linkage_corr(gics,corr)
        corr0=TheoryImpliedCorrelation()._link2corr(lnk0,corr.index)
        corr1=RiskEstimators().denoise_covariance(self, corr0, tn_relation, kde_bwidth=kde_bwidth)
        return corr1


    @staticmethod
    def corr_dist(corr0, corr1):
        """
        Calculates the correlation matrix distance proposed by Herdin and Bonek.
        The distance obtained measures the orthogonality between the considered
        correlation matrices. If the matrices are equal up to a scaling factor,
        the distance becomes zero and one if they are different to a maximum
        extent.
        This can be used to measure to which extent the TIC matrix has blended
        theory-implied views (tree structure of the elements) with empirical
        evidence (correlation matrix).
        :param corr0: (pd.dataframe) First correlation matrix
        :param corr1: (pd.dataframe) Second correlation matrix
        :return: (float) Correlation matrix distance
        """

        num=np.trace(np.dot(corr0,corr1))
        den=np.linalg.norm(corr0,ord='fro')
        den*=np.linalg.norm(corr1,ord='fro')
        cmd=1-num/den
        return cmd

    def _get_linkage_corr(self, tree, corr):
        """
        Fits the theoretical tree graph structure of the assets in a portfolio on the evidence
        presented by the empirical correlation matrix.
        The result is a binary tree (dendrogram) that sequentially clusters two elements
        together, while measuring how closely together the two elements are, until all elements are
        subsumed within the same cluster.
        This is the first step of the TIC algorithm.
        :param tree_struct: (pd.dataframe) The tree graph that represents the structure of the assets
        :param corr_matrix: (pd.dataframe) The empirical correlation matrix of the assets
        :return: (np.array) Linkage object that characterizes the dendrogram
        """

        if len(np.unique(tree.iloc[:,-1]))>1:tree['All']=0 # add top level
        lnk0=np.empty(shape=(0,4))
        lvls=[[tree.columns[i-1],tree.columns[i]] for i in range(1,tree.shape[1])]
        dist0=((1-corr)/2.)**.5 # distance matrix
        items0=dist0.index.tolist() # map lnk0 to dist0
        for cols in lvls:
            grps=tree[cols].drop_duplicates(cols[0]).set_index(cols[0]).groupby(cols[1])
            for cat,items1 in grps:
                items1=items1.index.tolist()
                if len(items1)==1: # single item: rename
                    items0[items0.index(items1[0])]=cat
                    dist0=dist0.rename({items1[0]:cat},axis=0)
                    dist0=dist0.rename({items1[0]:cat},axis=1)
                    continue
                dist1=dist0.loc[items1,items1]
                lnk1=sch.linkage(ssd.squareform(dist1,force='tovector',
                    checks=(not np.allclose(dist1,dist1.T))),
                    optimal_ordering=True) # cluster that cat
                lnk_=TheoryImpliedCorrelation()._link_clusters(lnk0,lnk1,items0,items1)
                lnk0=np.append(lnk0,lnk_,axis=0)
                items0+=range(len(items0),len(items0)+len(lnk_))
                dist0=TheoryImpliedCorrelation()._update_dist(dist0,lnk0,lnk_,items0)
                # Rename last cluster for next level
                items0[-1]=cat
                dist0.columns=dist0.columns[:-1].tolist()+[cat]
                dist0.index=dist0.columns
        lnk0=np.array(map(tuple,lnk0),dtype=[('i0',int),('i1',int), \
            ('dist',float),('num',int)])
        return lnk0

    @staticmethod
    def _link_clusters(lnk0, lnk1, items0, items1):
        """
        Transforms linkage object from local local_linkage (based on dist1) into global global_linkage (based on dist0)
        Consists of changes of names for the elements in clusters and change of the number of
        basic elements (atoms) contained inside a cluster. This is done to take into account the
        already existing links.
        :param global_linkage: (np.array) Global linkage object (previous links)
        :param local_linkage: (np.array) Local linkage object (containing grouped elements and not global ones)
        :param global_elements: (list) List of names for all elements (global)
        :param grouped_elements: (list) List of grouped elements (local)
        :return: (np.array) Local linkage object changed to global one
        """

        # transform partial link1 (based on dist1) into global link0 (based on dist0)
        nAtoms=len(items0)-lnk0.shape[0]
        lnk_=lnk1.copy()
        for i in range(lnk_.shape[0]):
            i3=0
            for j in range(2):
                if lnk_[i,j]<len(items1):
                    lnk_[i,j]=items0.index(items1[int(lnk_[i,j])])
            else:
                lnk_[i,j]+=-len(items1)+len(items0)
            # update number of items
            if lnk_[i,j]<nAtoms:i3+=1
            else:
                if lnk_[i,j]-nAtoms<lnk0.shape[0]:
                    i3+=lnk0[int(lnk_[i,j])-nAtoms,3]
                else:
                    i3+=lnk_[int(lnk_[i,j])-len(items0),3]
            lnk_[i,3]=i3
        return lnk_


    @staticmethod
    def _update_dist(dist0, lnk0, lnk_, items0, criterion=None):
        """
        Updates the general distance matrix to take the new clusters into account
        Replaces the elements added to the new clusters with these clusters as elements.
        Requires the recalculation of the distance matrix to determine the distance from
        new clusters to other elements.
        A criterion function may be given for calculation of the new distances from a new cluster to other
        elements based on the distances of elements included in a cluster. The default method is the weighted
        average of distances based on the number of atoms in each of the two elements.
        :param distance_matrix: (pd.dataframe) Previous distance matrix
        :param global_linkage: (np.array) Global linkage object that includes new clusters
        :param local_linkage_tr: (np.array) Local linkage object transformed (global names of elements and atoms count)
        :param global_elements: (list) Global list with names of all elements
        :param criterion: (function) Function to apply to a dataframe of distances to adjust them
        :return: (np.array) Updated distance matrix
        """

        # expand dist0 to incorporate newly created clusters
        nAtoms=len(items0)-lnk0.shape[0]
        newItems=items0[-lnk_.shape[0]:]
        for i in range(lnk_.shape[0]):
            i0,i1=items0[int(lnk_[i,0])],items0[int(lnk_[i,1])]
            if criterion is None:
                if lnk_[i,0]<nAtoms:w0=1.
                else:w0=lnk0[int(lnk_[i,0])-nAtoms,3]
                if lnk_[i,1]<nAtoms:w1=1.
                else:w1=lnk0[int(lnk_[i,1])-nAtoms,3]
                dist1=(dist0[i0]*w0+dist0[i1]*w1)/(w0+w1)
            else:dist1=criterion(dist0[[i0,i1]],axis=1) # linkage criterion
            dist0[newItems[i]]=dist1 # add column
            dist0.loc[newItems[i]]=dist1 # add row
            dist0.loc[newItems[i],newItems[i]]=0. # main diagonal
            dist0=dist0.drop([i0,i1],axis=0)
            dist0=dist0.drop([i0,i1],axis=1)
        return dist0

    @staticmethod
    def _get_atoms(lnk, item):
        """
        Getting the atoms included in an element from a linkage object
        Atoms are the basic assets in a portfolio and not clusters.
        :param linkage: (np.array) Global linkage object
        :param element: (int) Element id to get atoms from
        :return: (list) Set of atoms
        """

        # get all atoms included in an item
        anc=[item]
        while True:
            item_=max(anc)
            if item_<=lnk.shape[0]:break
            else:
                anc.remove(item_)
                anc.append(lnk['i0'][item_-lnk.shape[0]-1])
                anc.append(lnk['i1'][item_-lnk.shape[0]-1])
        return anc


    def _link2corr(self, lnk,lbls):
        """
        Derives a correlation matrix from the linkage object.
        Each cluster in the global linkage object is decomposed into two elements,
        which can be either atoms or other clusters. Then the off-diagonal correlation between two
        elements are calculated based on the distances between them.
        This is the second step of the TIC algorithm.
        :param linkage: (np.array) Global linkage object
        :param element_index: (pd.index) Names of elements used to calculate the linkage object
        :return: (pd.dataframe) Correlation matrix associated with linkage object
        """

        # derive the correl matrix associated with a given linkage matrix
        corr=pd.DataFrame(np.eye(lnk.shape[0]+1),index=lbls,columns=lbls,
                dtype=float)
        for i in range(lnk.shape[0]):
            x=TheoryImpliedCorrelation()._get_atoms(lnk,lnk['i0'][i])
            y=TheoryImpliedCorrelation()._get_atoms(lnk,lnk['i1'][i])
            corr.loc[lbls[x],lbls[y]]=1-2*lnk['dist'][i]**2 # off-diagonal values
            corr.loc[lbls[y],lbls[x]]=1-2*lnk['dist'][i]**2 # symmetry
        return corr