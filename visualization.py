# visualization.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class MultimodalVisualizer:
    def __init__(self, data=None):
        self.data = data
    
    def plot_party_votes(self, top_n=5):
        if self.data is None or 'party' not in self.data.columns or 'valid_votes' not in self.data.columns:
            return None
        try:
            self.data['valid_votes'] = pd.to_numeric(self.data['valid_votes'], errors='coerce')
            party_data = self.data.groupby('party')['valid_votes'].sum().reset_index()
            party_data = party_data.sort_values('valid_votes', ascending=False).head(top_n)
            fig = px.bar(party_data, x='party', y='valid_votes', title="Top Parties by Votes")
            return fig
        except:
            return None
    
    def plot_regional_distribution(self):
        if self.data is None or 'region' not in self.data.columns or 'valid_votes' not in self.data.columns:
            return None
        try:
            self.data['valid_votes'] = pd.to_numeric(self.data['valid_votes'], errors='coerce')
            region_data = self.data.groupby('region')['valid_votes'].sum().reset_index()
            fig = px.pie(region_data, names='region', values='valid_votes', title="Vote Distribution by Region")
            return fig
        except:
            return None
    
    def plot_party_comparison_by_region(self):
        if self.data is None or 'region' not in self.data.columns or 'party' not in self.data.columns or 'valid_votes' not in self.data.columns:
            return None
        try:
            self.data['valid_votes'] = pd.to_numeric(self.data['valid_votes'], errors='coerce')
            region_party_data = self.data.groupby(['region', 'party'])['valid_votes'].sum().reset_index()
            fig = px.bar(region_party_data, x='region', y='valid_votes', color='party', 
                        title="Party Votes by Region", barmode='group')
            return fig
        except:
            return None
    
    def plot_voter_turnout(self):
        if self.data is None or 'region' not in self.data.columns or 'valid_votes' not in self.data.columns or 'registered_voters' not in self.data.columns:
            return None
        try:
            self.data['valid_votes'] = pd.to_numeric(self.data['valid_votes'], errors='coerce')
            self.data['registered_voters'] = pd.to_numeric(self.data['registered_voters'], errors='coerce')
            turnout_data = self.data.groupby('region').agg({
                'valid_votes': 'sum',
                'registered_voters': 'sum'
            }).reset_index()
            turnout_data['turnout'] = turnout_data['valid_votes'] / turnout_data['registered_voters'] * 100
            fig = px.bar(turnout_data, x='region', y='turnout', title="Voter Turnout by Region (%)")
            return fig
        except:
            return None