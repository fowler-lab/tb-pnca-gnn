import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np
import torch
import sbmlcore
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch_geometric.data import Data, HeteroData
from src.model_helpers import convert_3letter_to_sequence

class ProteinGraph():
    """
    Class for BlaC protein graph construction.
    """
    
    def __init__(
        self, 
        pdb: str, 
        lig_resname: str, 
        self_loops: bool, 
        cutoff_distance = 5.6,
        end_length = False
        ):
        
        self.pdb = pdb
        self.cutoff_distance = cutoff_distance
        self.lig_resname = lig_resname
        self.self_loops = self_loops
        
        self.lig_selection = f'resname {self.lig_resname} and chainID A'
        self.end_length = end_length
        
        self.nodes = self._get_protein_struct_nodes()
        self.edge_index, self.d_array = self._get_protein_struct_edges(self.nodes.center_of_mass(compound='residues'))
        self.edge_dists = self._gen_edge_dists(self.edge_index, self.d_array)
        
        assert len(self.nodes.residues) > max(torch.cat([self.edge_index[0], self.edge_index[1]])), "Edge index out of bounds"
    
    def _get_protein_struct_nodes(self):
        protein_structure = mda.Universe(self.pdb)
        nodes = protein_structure.select_atoms('protein and chainID A')
        
        # restrict nodes to BlaC genbank sequence (hardcoded)
        first_res = nodes.residues.resids[:-2][0]
        last_res = nodes.residues.resids[:-2][-1]
        nodes = nodes.select_atoms(f'resid {first_res}:{last_res}')
        
        return nodes
    
    def _get_protein_struct_edges(self, node1_positions, node2_positions=None):
        
        if node2_positions is None:
            node2_positions = node1_positions
        
        d_array = distance_array(node1_positions, node2_positions, backend='OpenMP')

        if self.self_loops:
            mask = d_array < self.cutoff_distance
        else:
            mask = (0 < d_array) & (d_array < self.cutoff_distance)

        # convert to correct form for PyTorch Geometric
        edges = np.argwhere(mask)
        edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
        
        if not self.self_loops:
            for s,t in zip(edge_index[0], edge_index[1]):
                assert s != t, f"Self loop found but self_loops was set to {self.self_loops}: ({s},{t})"
        
        return edge_index, d_array
    
    def _gen_edge_dists(self, edge_index, d_array):
        edge_dists = torch.tensor([])
    
        for edge in range(len(edge_index[0])):
            i = edge_index[0][edge]
            j = edge_index[1][edge]
            
            edge_distance = d_array[i][j]
            
            edge_dists = torch.cat((edge_dists, torch.tensor([edge_distance])), 0)
        
        edge_dists = edge_dists.type(torch.float32)
        
        return edge_dists
    
    def build_resids(self, sequence):
        resids = []
        counter = 1
        for i in sequence:
            resids.append(('A',i,counter))
            counter += 1
        return np.array(resids, dtype = object)
    
    def gen_dist_features(self, wt_seq: str) -> pd.DataFrame:
        resids = self.build_resids(wt_seq)
        res_df = pd.DataFrame(resids, columns=['segid', 'amino_acid', 'resid'])
        # hardcoded for BlaC
        dists_dataset = sbmlcore.FeatureDataset(res_df,species='M. tuberculosis', gene='blaC', protein='BlaC')
        dist = sbmlcore.StructuralDistances(self.pdb, self.lig_selection, 'CLAV_dist',dataset_type='amino_acid', infer_masses=False, offsets={'A': -27, 'B': -27})
        # fixed extra residue in pdb
        dist.results = dist.results.drop(236)
        dist.results.reset_index(drop=True, inplace=True)
        
        dists_dataset.add_feature(dist)

        # Botch the distance for 1st res (A) to match the distance for the first M 
        dists_dataset.df.loc[0, 'CLAV_dist'] = dist.results.loc[0, 'CLAV_dist']
        dists_dataset.df.set_index(['segid','resid'],inplace=True)
        
        return dists_dataset.df
    

    def build_feature_dataframe(self, resids, dists_dataset=None):
        
        df = pd.DataFrame(resids, columns=['segid', 'amino_acid', 'resid'])

        features = sbmlcore.FeatureDataset(df,species='M. tuberculosis', gene='blaC', protein='BlaC')

        v = sbmlcore.AminoAcidVolume()
        h = sbmlcore.AminoAcidHydropathyWimleyWhite()
        mw = sbmlcore.AminoAcidMW()
        p = sbmlcore.AminoAcidPi()
        kd = sbmlcore.AminoAcidHydropathyKyteDoolittle()
        ha = sbmlcore.HBondAcceptors()
        hd = sbmlcore.HBondDonors()
        r = sbmlcore.SideChainRings()

        features.add_feature([v, h, mw, p, kd, ha, hd, r])

        for col in ['volume', 'hydropathy_WW', 'MW', 'Pi', 'hydropathy_KD', 'h_acceptors', 'h_donors', 'rings']:
            features.df[col] = features.df[col].astype('float') 

        # Add CLAV_dist to the features
        if dists_dataset is not None:
            features.df.set_index(['segid', 'resid'], inplace=True)
            features.df = features.df.join(dists_dataset[['CLAV_dist']], on=['segid', 'resid'], how='inner')
            features.df.reset_index(inplace=True)
            features.df['CLAV_dist'] = features.df['CLAV_dist'].astype('float') 
        
        features.df = features.df.sort_values(by='resid', ascending=True)
         
        x = np.array(features.df)[:,3:]
        # transformer = MinMaxScaler().fit(x)
        # x = transformer.transform(x)
        x = torch.tensor(x.tolist(), dtype=torch.float)

        return x
    
    def calc_edge_weights(self, function, edge_dist, lambda_param=None):
        
        if function == "dist":
            ews = edge_dist
        elif function == "1-(dist/cutoff)":
            ews = 1 - (edge_dist / self.cutoff_distance)
        elif function == "1/dist":
            if self.self_loops:
                raise ValueError("Cannot use '1/dist' edge weights with self loops")
            ews = 1 / edge_dist
        elif function == "exp":
            if lambda_param is None:
                raise ValueError("Lambda parameter must be provided for exponentially decaying edge weights")
            ews = np.exp(-lambda_param * edge_dist)
        elif function == "none":
            ews = None
        else:
            raise ValueError(f"Invalid edge weight function {function}, choose from 'dist', '1-(dist/cutoff)', '1/dist', 'exp', 'none'")
        
        return ews
    
    def process_edge_weights(self, edge_weights):
        
        edge_weights = np.array(edge_weights).reshape(-1,1)
        transformer = MinMaxScaler().fit(edge_weights)
        edge_weights = transformer.transform(edge_weights)
        edge_weights = torch.tensor(edge_weights)
            
        return edge_weights
    
    def gen_dataset(
        self, 
        wt_seq, 
        sequences: pd.DataFrame,
        edge_weights: str,
        lambda_param: float = None,
        normalise: bool = True
        ):
        """_summary_

        Args:
            wt_seq (_type_): _description_
            sequences (pd.DataFrame): _description_
            edge_weights (_type_): "dist", "1-(dist/cutoff)", "1/dist", "none"
            normalise (bool): When True, edge weights are normalised to the range [0,1]
        """
        
        dataset = []
        
        ews = self.calc_edge_weights(edge_weights, self.edge_dists, lambda_param=lambda_param)
        
        if normalise and edge_weights != 'none':
            ews = self.process_edge_weights(ews)
        
        # dists_df = self.gen_dist_features(wt_seq)
            
        for idx,row in tqdm(sequences.iterrows(), total=len(sequences), disable=(len(sequences) == 1)):
            
            if self.end_length:
                resids = self.build_resids(row.allele[:self.end_length])
            else:
                resids = self.build_resids(row.allele)
            
            x = self.build_feature_dataframe(resids, 
                                            #  dists_dataset = dists_df
                                             )
            
            y = torch.tensor(1 if row.phenotype_label == "R" else 0)
            
            data = Data(x=x, 
                        edge_index=self.edge_index,
                        edge_attr=ews, 
                        y=y, 
                        # device="cuda" if torch.cuda.is_available() else "mps"
                        )

            # if torch.cuda.is_available():
            #     data = data.to('cuda')
            
            assert data.validate(raise_on_error=True)

            dataset.append(data)
        
        self.dataset = dataset
        
    def attach_snap2(self, df):
    
        assert "snap2_score" not in df.columns, "Snap2 already added"
        
        assert (
                "mutation" in df.columns
            ), "Passed dataframe must contain a column called mutation"
        
        snap2 = sbmlcore.SNAP2('../data/features/3pl1-snap2-with-segids.csv', offsets={'A': 0})
        
        df = pd.merge(df, snap2.results[['mutation', 'snap2_score']], on='mutation', how='left')
        df['snap2_score'] = df['snap2_score'].fillna(-100)
        
        return df
    
    def attach_deepddg(self, df):
        
        assert "deep_ddG" not in df.columns, "DeepDDG already added"
        
        deepddg = sbmlcore.DeepDDG('../data/features/3pl1.ddg')
        
        df = pd.merge(df, deepddg.results, 
                            left_on=['segid', 'ref', 'resid', 'amino_acid'], 
                            right_on=['segid', 'ref_amino_acid', 'resid', 'alt_amino_acid'],
                            how='left')
        df.drop(columns = ['ref_amino_acid', 'alt_amino_acid'], inplace=True)
        df['deep_ddG'] = df['deep_ddG'].fillna(0.0)
        
        return df
    
    def attach_rasp(self, df):
    
        assert "rasp_score_ml" not in df.columns, "RaSP already added"

        rasp = sbmlcore.RaSP('../data/features/cavity_pred_3PL1_A.csv')
        
        df = pd.merge(df, rasp.results, 
                            left_on=['segid', 'ref', 'resid', 'amino_acid'], 
                            right_on=['segid', 'ref_amino_acid', 'resid', 'alt_amino_acid'],
                            how='left')
        df.drop(columns = ['ref_amino_acid', 'alt_amino_acid', 'rasp_wt_nlf', 'rasp_mt_nlf', 'rasp_score_ml_fermi'], inplace=True)

        df['rasp_score_ml'] = np.where(
            df['mutation'].isna(),
            0.0,
            df['rasp_score_ml']
        )
        
        return df
    
    def attach_mapp(self, df):
    
        assert "mapp_score" not in df.columns, "MAPP already added"    
        
        mapp_df = pd.read_csv('../data/features/3pl1-mapp_scores.csv')
        mapp = pd.melt(mapp_df, id_vars='Position',value_vars=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        mapp.rename(columns={'Position': 'codon', 'variable': 'alt_amino_acid', 'value': 'mapp_score'}, inplace=True)

        df = pd.merge(df, mapp, 
                            left_on=['resid', 'amino_acid'], 
                            right_on=['codon','alt_amino_acid'],
                            how='left')
        df.drop(columns = ['codon', 'alt_amino_acid'], inplace=True)


        df['mapp_score'] = np.where(
            df['mutation'].isna(),
            0.0,
            df['mapp_score']
        )
        
        return df
    
    def add_mutation_features(self, features_df):
        
        wt_seq = 'MRALIIVDVQNDFCEGGSLAVTGGAALARAISDYLAEAADYHHVVATKDFHIDPGDHFSGTPDYSSSWPPHCVSGTPGADFHPSLDTSAIEAVFYKGAYTGAYSGFEGVDENGTPLLNWLRQRGVDEVDVVGIATDHCVRQTAEDAVRNGLATRVLVDLTAGVSADTTVAALEEMRTASVELVCS'
        
        features_df = pd.concat([features_df, pd.DataFrame(list(wt_seq), columns=['ref'])], axis=1)
        
        features_df['mutation'] = features_df.apply(
            lambda row: f"{row['ref']}{row['resid']}{row['amino_acid']}" 
            if row['amino_acid'] != row['ref'] else None, 
            axis=1
            )
        
        features_df = self.attach_snap2(features_df)
        features_df = self.attach_deepddg(features_df)
        features_df = self.attach_rasp(features_df)
        features_df = self.attach_mapp(features_df)
        
        features_df.drop(columns=['ref', 'mutation'], inplace=True)
        
        return features_df
        
class pncaGraph(ProteinGraph):
    """
    Class for PncA protein graph construction. Inherits from ProteinGraph (BlaC graph class).
    """
    def __init__(
        self, 
        pdb: str,
        lig_resname: str, 
        self_loops: bool,
        # lig_pdb: str = None, #! temp arg to use coordinates for CLAV bound to WT 
        cutoff_distance = 5.6
        ):
        
        self.end_length = 185
        
        super().__init__(pdb, lig_resname, self_loops, cutoff_distance, self.end_length)
        
        # # temp code
        # if lig_pdb is None:
        #     self.lig_pdb = self.pdb
        # else:
        #     self.lig_pdb = lig_pdb
        # # temp code
        
        self.lig_selection = f'resname {self.lig_resname}'
            
    def _get_protein_struct_nodes(self):
        protein_structure = mda.Universe(self.pdb)
        nodes = protein_structure.select_atoms('protein and not name C N O')

        # restrict nodes to first 185 Pnca residues (last one not included in -PZA bound pdb)
        first_res = nodes.residues.resids[0]
        last_res = nodes.residues.resids[self.end_length-1]
        nodes = nodes.select_atoms(f'resid {first_res}:{last_res}')
        
        return nodes
    
    def gen_dist_features(self, wt_seq: str) -> pd.DataFrame:
        # resids = self.build_resids(wt_seq)
        seq = convert_3letter_to_sequence(self.nodes.residues.resnames)
        resids = self.build_resids(seq)
        
        res_df = pd.DataFrame(resids, columns=['segid', 'amino_acid', 'resid'])

        dists_dataset = sbmlcore.FeatureDataset(res_df,species='M. tuberculosis', gene='pncA', protein='PncA')
     
        dist = sbmlcore.StructuralDistances(self.pdb, self.lig_selection, 'PZA_dist',dataset_type='amino_acid', infer_masses=False)
        
        dists_dataset.add_feature(dist)
        
        stride = sbmlcore.Stride(self.pdb, dataset_type='amino_acid')
        dists_dataset.add_feature(stride)
        
        dists_dataset.df.set_index(['segid','resid'],inplace=True)
        
        return dists_dataset.df
    
    def build_feature_dataframe(self, 
                                resids, 
                                # dists_dataset=None
                                ):
        
        df = pd.DataFrame(resids, columns=['segid', 'amino_acid', 'resid'])

        features = sbmlcore.FeatureDataset(df,species='M. tuberculosis', gene='pncA', protein='PncA')

        v = sbmlcore.AminoAcidVolume()
        h = sbmlcore.AminoAcidHydropathyWimleyWhite()
        mw = sbmlcore.AminoAcidMW()
        p = sbmlcore.AminoAcidPi()
        kd = sbmlcore.AminoAcidHydropathyKyteDoolittle()
        ha = sbmlcore.HBondAcceptors()
        hd = sbmlcore.HBondDonors()
        r = sbmlcore.SideChainRings()
        dist = sbmlcore.StructuralDistances(self.pdb, self.lig_selection, 'PZA_dist',dataset_type='amino_acid', infer_masses=False)
        stride = sbmlcore.Stride(self.pdb, dataset_type='amino_acid')
        depth = sbmlcore.ResidueDepth(self.pdb, segids=['A'])
        fe_dist = sbmlcore.StructuralDistances(self.pdb, 'resname FE2', 'FE2_dist', dataset_type='amino_acid', infer_masses=False)
            
        features.add_feature([v, h, mw, p, kd, ha, hd, r, dist, stride, depth, fe_dist])

        # for col in ['volume', 'hydropathy_WW', 'MW', 'Pi', 'hydropathy_KD', 'h_acceptors', 'h_donors', 'rings']:
        #     features.df[col] = features.df[col].astype('float') 

        features.df.drop(columns=['n_hbond_acceptors', 'n_hbond_donors'], inplace=True)

        for col in features.df.columns:
            if is_numeric_dtype(features.df[col]) and not is_bool_dtype(features.df[col]) and col != 'resid':
                features.df[col] = features.df[col].astype('float') 
            elif col not in ['resid', 'segid', 'amino_acid']:
                features.df.drop(columns=col, inplace=True)
                
        # # Add PZA_dist to the features
        # if dists_dataset is not None:
        #         features.df.set_index(['segid', 'resid'], inplace=True)
        #         features.df = features.df.join(dists_dataset[['PZA_dist', 'phi', 'psi', 'residue_sasa']], on=['segid', 'resid'], how='inner')
        #         features.df.reset_index(inplace=True)
        #         features.df['PZA_dist'] = features.df['PZA_dist'].astype('float') 
        
        features.df = self.add_mutation_features(features.df)
        
        features.df = features.df.sort_values(by='resid', ascending=True)
        
        x = np.array(features.df)[:,3:]
        # transformer = MinMaxScaler().fit(x)
        # x = transformer.transform(x)
        x = torch.tensor(x.tolist(), dtype=torch.float)

        return x
 
    
    
class ProteinLigandGraph(ProteinGraph):
    """
    Class for BlaC protein-ligand graph construction for HeteroGCN. Inherits from ProteinGraph (BlaC graph class).
    """
    def __init__(
        self, 
        pdb: str, 
        lig_resname: str, 
        self_loops: bool, 
        cutoff_distance = 5.6
        ):
        
        super().__init__(pdb, lig_resname, self_loops, cutoff_distance)
        
        self.lig_node_positions = self._get_ligand_nodes()
        self.edge_index_dict = self._gen_edge_index_dict()
        self.edge_dists_dict = self._gen_edge_dists_dict()
        
        
    def _get_ligand_nodes(self):
        assert self.lig_resname == 'ISS' and self.pdb[-8:] == '6h2c.pdb', "Only ISS ligand in 6h2c.pdb currently implemented"
        
        protein_structure = mda.Universe(self.pdb)
        lig_nodes = protein_structure.select_atoms(f'resname {self.lig_resname} and chainID A')
        
        # bits determined by BRICS decomposition
        bit1 = protein_structure.select_atoms(f'index {lig_nodes.atoms.indices[1]}:{lig_nodes.atoms.indices[4]}')
        bit2 = protein_structure.select_atoms(f'index {lig_nodes.atoms.indices[5]}:{lig_nodes.atoms.indices[-1]}')
        
        assert set(bit1.atoms.indices).isdisjoint(set(bit2.atoms.indices)), 'BRICS decomposed fragments overlap'

        for atom in bit1 + bit2:
            assert atom.resname == self.lig_resname, f'Index of ligand atoms does not match with ligand resname ({self.lig_resname})'
        
        bit1_pos = bit1.center_of_geometry(compound='group')
        bit2_pos = bit2.center_of_geometry(compound='group')
        
        lig_positions = np.stack((bit1_pos, bit2_pos))
        
        return lig_positions

    def _gen_edge_index_dict(self):
        
        res_lig_edge_index,res_lig_d_array = self._get_protein_struct_edges(self.nodes.center_of_mass(compound='residues'), self.lig_node_positions)
        lig_res_edge_index, lig_res_d_array = self._get_protein_struct_edges(self.lig_node_positions, self.nodes.center_of_mass(compound='residues'))
        lig_lig_edge_index, lig_lig_d_array = self._get_protein_struct_edges(self.lig_node_positions)
        
        return {('residue', 'links', 'residue'): {'edge_index': self.edge_index, 'd_array': self.d_array},
                ('residue', 'links', 'ligand'): {'edge_index': res_lig_edge_index, 'd_array': res_lig_d_array},
                ('ligand', 'links', 'residue'): {'edge_index': lig_res_edge_index, 'd_array': lig_res_d_array},
                ('ligand', 'links', 'ligand'): {'edge_index': lig_lig_edge_index, 'd_array': lig_lig_d_array}}
        
    def _gen_edge_dists_dict(self):
        edge_dists_dict = {}
        
        for edge_type in self.edge_index_dict:
            edge_index = self.edge_index_dict[edge_type]['edge_index']
            d_array = self.edge_index_dict[edge_type]['d_array']
            
            edge_dists = self._gen_edge_dists(edge_index, d_array)
            
            edge_dists_dict[edge_type] = edge_dists
        
        return edge_dists_dict
    
    def gen_dataset(
        self, 
        wt_seq, 
        sequences: pd.DataFrame,
        edge_weights: str,
        normalise: bool = False
        ):
        """_summary_

        Args:
            wt_seq (_type_): _description_
            sequences (pd.DataFrame): _description_
            edge_weights (_type_): "dist", "1-(dist/cutoff)", "1/dist", "none"
            normalise (bool): When True, edge weights are normalised to the range [0,1]
        """
        
        dataset = []
        
        edge_weights_dict = {}
        
        for edge_type in self.edge_dists_dict:
            edge_weights_dict[edge_type] = self.calc_edge_weights(edge_weights, self.edge_dists_dict[edge_type])
                 
        # Concatenate all edge weights and normalize them together          
        if normalise and edge_weights != 'none':
            all_edge_dists = torch.cat(list(edge_weights_dict.values()))
            all_edge_weights = np.array(all_edge_dists).reshape(-1,1)
            transformer = MinMaxScaler().fit(all_edge_weights)
            all_edge_weights = transformer.transform(all_edge_weights)
            all_edge_weights = torch.tensor(all_edge_weights)
            
            # reassign normalized weights to edge_dists_dict
            last_index = 0
            for edge_type in edge_weights_dict:
                edge_weights_dict[edge_type] = all_edge_weights[last_index:last_index+len(edge_weights_dict[edge_type])]
            
            
        dists_df = self.gen_dist_features(wt_seq)
            
        for idx,row in tqdm(sequences.iterrows(), total=len(sequences)):
            
            resids = self.build_resids(row.allele)
            
            x = self.build_feature_dataframe(resids, dists_dataset = dists_df)
            
            y = torch.tensor(1 if row.phenotype_label == "R" else 0)
            
            data = HeteroData()
            data['residue'].x = x
            data['ligand'].x = torch.ones((2, 9))
            
            data.y = y
            
            for edge_type in self.edge_index_dict:
                data[edge_type].edge_index = self.edge_index_dict[edge_type]['edge_index']
                data[edge_type].edge_attr = edge_weights_dict[edge_type]
            
            assert data.validate(raise_on_error=True)

            dataset.append(data)
            
        self.dataset = dataset