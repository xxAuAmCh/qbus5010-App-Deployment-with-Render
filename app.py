!pip install dash-bootstrap-components
!pip install folium

import pandas as pd
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import KDTree
import folium
import json

overall_year_p = pd.read_csv('d_00_21_p.csv')
overall_age_21p = pd.read_csv('d_a_21_p.csv')

type1_years_p = pd.read_csv('t1_cay_14_21_p.csv')
type1_age_21p = pd.read_csv('t1_cay_21_p.csv')
type2_years_p = pd.read_csv('t2_00_21_p.csv')
type2_age_21p = pd.read_csv('t2_a_21_p.csv')

overall_year_i = pd.read_csv('d_00_21_i.csv')
overall_age_21i = pd.read_csv('d_a_21_i.csv')

type1_years_i = pd.read_csv('t1_00_21_i.csv')
type1_age_21i = pd.read_csv('t1_a_21_i.csv')
type2_years_i = pd.read_csv('t2_00_21_i.csv')
type2_age_21i = pd.read_csv('t2_a_21_i.csv')

total_diabetic = "1.3 m"
rate_of_diabetes = "5.3%"
male_rate = "5.8%"
female_rate = "4.9%"

data = {'Diabetes Type': ['Type 1 diabetes', 'Type 2 diabetes', 'Type not known'],
        'Percentage': [9.6, 87.6, 2.8],}
pie_d = pd.DataFrame(data)
pie_d['Diabetes Type'] = pd.Categorical(pie_d['Diabetes Type'],
                                        categories = ["Type 1 diabetes", "Type 2 diabetes", "Type not known"],
                                        ordered = True)


fig_p = px.pie(pie_d,
               names = "Diabetes Type",
               values = "Percentage",
               hole = 0.4,
               color = "Diabetes Type",
               category_orders = {'Diabetes Type': ['Type 1 diabetes',
                                                    'Type 2 diabetes',
                                                    'Type not known']},
               color_discrete_map = {'Type 1 diabetes': '#E06ACB',
                                     'Type 2 diabetes': '#235F94',
                                     'Type not known': '#F5C014'})
fig_p.update_layout(showlegend = True,
                    title = dict(text = "Diabetes Type",
                                 x = 0.5,
                                 xanchor = 'center'),
                    legend = dict(orientation = "h",
                                  y = -0.2,
                                  x = 0.5,
                                  xanchor = 'center',
                                  font = dict(size = 10),
                                  title = dict(text = '')),
                    margin = dict(t = 40, b = 0, l = 0, r = 0),
                    height = 400)

postcode_data = pd.read_csv("au_postcodes.csv")
postcode_summary = pd.read_csv("Postcode Summary.csv")
merged_data = pd.merge(postcode_data, postcode_summary, how='inner', left_on='postcode', right_on='Postcode')

merged_data['% Population'] = merged_data['% Population'].fillna('No Data Available')

merged_data['% Population'] = pd.to_numeric(merged_data['% Population'].str.replace('%', ''), errors='coerce')

filtered_data = merged_data[merged_data['% Population'].notna()]
geojson_files = [
    'suburb-10-act.geojson',
    'suburb-10-nsw.geojson',
    'suburb-10-nt.geojson',
    'suburb-10-qld.geojson',
    'suburb-10-tas.geojson',
    'suburb-10-vic.geojson',
    'suburb-10-wa.geojson',
    'suburb-10-sa.geojson'
]

def add_population_to_geojson_using_kdtree(geojson_file, merged_data):
    with open(geojson_file) as f:
        geojson_data = json.load(f)

    lat_lon_array = merged_data[['latitude', 'longitude']].values
    kdtree = KDTree(lat_lon_array)

    postcode_dict = {(row['latitude'], row['longitude']): row['postcode'] for _, row in merged_data.iterrows()}
    population_dict = {(row['latitude'], row['longitude']): row['% Population'] for _, row in merged_data.iterrows()}

    for feature in geojson_data['features']:
        coords = feature['geometry']['coordinates'][0][0]
        centroid = (coords[1], coords[0])

        distance, index = kdtree.query(centroid)
        nearest_lat_lon = tuple(lat_lon_array[index])

        feature['properties']['postcode'] = postcode_dict.get(nearest_lat_lon, "No Data Available")
        feature['properties']['% Population'] = population_dict.get(nearest_lat_lon, "No Data Available")

    return geojson_data

modified_geojson_files = [
    add_population_to_geojson_using_kdtree(geojson_file, filtered_data)
    for geojson_file in geojson_files
]


def style_function(feature):
    population_percentage = feature['properties'].get('% Population', "No Data Available")

    if not isinstance(population_percentage, (int, float)):
        color = 'gray'
    elif population_percentage <= 2:
        color = 'green'
    elif population_percentage <= 4:
        color = 'greenyellow'
    elif population_percentage <= 6:
        color = 'yellow'
    elif population_percentage <= 8:
        color = 'orange'
    else:
        color = 'red'

    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.1,
        'fillOpacity': 0.6
    }

def add_legend_based_on_style(map_object):
    legend_html = '''
     <div style="position: fixed;
                 bottom: 50px; left: 50px; width: 180px; height: 220px;
                 background-color: white; z-index:1000; font-size:14px;
                 border:2px solid grey; padding: 10px;">
     <b>% Population Legend</b><br>
     <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> 0-2%<br>
     <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> 2-4%<br>
     <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> 4-6%<br>
     <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> 6-8%<br>
     <i style="background: darkred; width: 10px; height: 10px; display: inline-block;"></i> >8%<br>
     <i style="background: grey; width: 10px; height: 10px; display: inline-block;"></i> No Data
     </div>
    '''
    map_object.get_root().html.add_child(folium.Element(legend_html))

map_australia_with_style_legend = folium.Map(location=[-25.2744, 133.7751], zoom_start=4, tiles="OpenStreetMap")

for geojson_data in modified_geojson_files:
    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        name='Diabetes Proportion'
    ).add_to(map_australia_with_style_legend)

folium.LayerControl().add_to(map_australia_with_style_legend)

add_legend_based_on_style(map_australia_with_style_legend)

map_australia_with_style_legend.save('australian_diabetes_map_with_style_legend.html')

state_data = pd.read_csv('State Summary.csv')
state_data_filtered = state_data.iloc[:, :4]

state_data_filtered['Proportion'] = pd.to_numeric(state_data_filtered['Proportion'].str.replace('%', ''), errors='coerce')

with open('australian-states.min.geojson') as ff:
    geojson_data_states = json.load(ff)

proportion_dict_filtered = state_data_filtered.set_index('State')['Proportion'].to_dict()

for feature in geojson_data_states['features']:
    state_name = feature['properties']['STATE_NAME']
    feature['properties']['Proportion'] = proportion_dict_filtered.get(state_name, "No Data Available")

def style_function_for_proportion(feature):
    proportion_value = feature['properties'].get('Proportion', "No Data Available")

    if not isinstance(proportion_value, (int, float)):
        color = 'gray'
    elif proportion_value <= 2:
        color = 'green'
    elif proportion_value <= 4:
        color = 'greenyellow'
    elif proportion_value <= 6:
        color = 'yellow'
    elif proportion_value <= 8:
        color = 'orange'
    else:
        color = 'red'

    return {
        'fillColor': color,
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.6
    }

def add_legend_for_proportion(map_object):
    legend_html = '''
     <div style="position: fixed;
                 bottom: 50px; left: 50px; width: 180px; height: 220px;
                 background-color: white; z-index:1000; font-size:14px;
                 border:2px solid grey; padding: 10px;">
     <b>Proportion Legend</b><br>
     <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> 0-2%<br>
     <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> 2-4%<br>
     <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> 4-6%<br>
     <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> 6-8%<br>
     <i style="background: darkred; width: 10px; height: 10px; display: inline-block;"></i> >8%<br>
     <i style="background: grey; width: 10px; height: 10px; display: inline-block;"></i> No Data
     </div>
    '''
    map_object.get_root().html.add_child(folium.Element(legend_html))

map_proportion_with_legend = folium.Map(location=[-25.2744, 133.7751], zoom_start=4, tiles="OpenStreetMap")

folium.GeoJson(
    geojson_data_states,
    style_function=style_function_for_proportion,
    name='Proportion Distribution'
).add_to(map_proportion_with_legend)

folium.LayerControl().add_to(map_proportion_with_legend)

add_legend_for_proportion(map_proportion_with_legend)

map_proportion_with_legend.save('australian_proportion_state_filtered_map_with_legend.html')

postcode_map_path = 'australian_diabetes_map_with_style_legend.html'
state_map_path = 'australian_proportion_state_filtered_map_with_legend.html'

def load_map_content(file_path):
    try:
        with open(file_path, 'r') as file_C:
            return file_C.read()
    except Exception as e:
        return f"Error loading map: {str(e)}"

postcode_map_content = load_map_content(postcode_map_path)
state_map_content = load_map_content(state_map_path)

obesity = pd.read_csv('obesity.csv')

df_melted1 = obesity.iloc[:,0:7].melt(id_vars='Age Group (2-17)', var_name='Category', value_name='Percentage')
df_melted1['Group'] = df_melted1['Category'].apply(lambda x: 'Boys' if 'Boys' in x else 'Girls')

df_melted2 = obesity.iloc[:,7:].melt(id_vars='Age Group (Above 18)', var_name='Category', value_name='Percentage')
df_melted2['Group'] = df_melted2['Category'].apply(lambda x: 'Men' if 'Men' in x else 'Women')

colors = ['turquoise', 'lightseagreen', 'darkcyan', 'lightgreen', 'yellowgreen', 'darkgreen']

bmi = pd.read_csv('Proportion with Type 2 diabetes by Body Mass Index, 2017-18.csv')
bmi = bmi.dropna()

plt_E2 = px.bar(bmi, 
                x='Body Mass Index', 
                y='Persons aged 18 years and over (%)', 
                title='Proportion of Type 2 Diabetes by BMI', 
                color_discrete_sequence=['#166183'])
plt_E2.update_layout(
    yaxis_title_text='Proportion',
    xaxis_title_text='BMI(Body Mass Index)', 
    paper_bgcolor='white',
    plot_bgcolor='white'
)
plt_E2.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
plt_E2.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)

def created_time_series_fig_number(df):
    fig = px.line(df,
                  x = 'year',
                  y = ['males', 'females', 'all'],
                  color_discrete_map = {'males': 'cornflowerblue',
                                        'females': 'olivedrab',
                                        'all': 'mediumpurple'})
    fig.update_layout(paper_bgcolor = 'white',
                      plot_bgcolor = 'white')
    fig.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    return fig

def created_time_series_fig_percent(df):
    fig = px.line(df,
                  x = 'year',
                  y = ['per_cent_males', 'per_cent_females', 'per_cent_persons'],
                  color_discrete_map = {'per_cent_males': 'cornflowerblue',
                                        'per_cent_females': 'olivedrab',
                                        'per_cent_persons': 'mediumpurple'})
    fig.update_layout(paper_bgcolor = 'white',
                      plot_bgcolor = 'white')
    fig.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    return fig

def created_time_series_fig_100k(df):
    fig = px.line(df,
                  x = 'year',
                  y = ['per_males', 'per_females', 'per_persons'],
                  color_discrete_map = {'per_males': 'cornflowerblue',
                                        'per_females': 'olivedrab',
                                        'per_persons': 'mediumpurple'})
    fig.update_layout(paper_bgcolor = 'white',
                      plot_bgcolor = 'white')
    fig.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    return fig

def create_fig(df):
    fig = go.Figure()
    max_value = max(df['males'].max(), df['females'].max())
    min_value = min(df['males'].min(), df['females'].min())
    tickvals = [-max_value, -max_value/2, 0, max_value/2, max_value]
    ticktext = ['{}'.format(int(abs(max_value))), '{}'.format(int(abs(max_value/2))),
                '0', '{}'.format(int(max_value/2)), '{}'.format(int(max_value))]
    fig.add_bar(x = -df['males'],
                y = df['age_group'],
                orientation = 'h',
                name = 'Males',
                marker_color = '#307B91',
                hovertemplate = '%{y}: %{customdata} males',
                customdata = df['males'])

    fig.add_bar(x = df['females'],
                y = df['age_group'],
                orientation = 'h',
                name = 'Females',
                marker_color = '#7EA652',
                hovertemplate = '%{y}: %{x} females')
    fig.update_layout(barmode = 'overlay',
                      xaxis_title = 'Population',
                      yaxis_title = 'Age Group',
                      legend_title_text = 'Gender',
                      xaxis = dict(tickmode = 'array',
                                   tickvals = tickvals,
                                   ticktext = ticktext,
                                   range = [-1.1 * max_value, 1.1 * max_value]),
                     paper_bgcolor = 'white',
                     plot_bgcolor = 'white')
    fig.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', showgrid=False)
    return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dbc.Row([
        html.H1('Diabetes in Australia',
                style={'backgroundColor': '#297D95',
                       'color': 'white',
                       'padding': '10px',
                       'text-align': 'center'})
    ],
    style={"width": "100%", "margin-bottom": "20px"}),

    dbc.Row([
    dbc.Col([
    html.Div([
        html.Div([
            html.Div([html.H4("Total diabetic", style={'margin-bottom': '0'}),
                      html.H1(total_diabetic, style={'margin-top': '5px'})],
                     style={"border": "2px solid #eee",
                            "padding": "2px",
                            "margin-bottom": "10px",
                            "text-align": "center"}),

            html.Div([html.H4("Rate of diabetes", style={'margin-bottom': '0'}),
                      html.H1(rate_of_diabetes, style={'margin-top': '5px'})],
                     style={"border": "2px solid #eee",
                            "padding": "2px",
                            "margin-bottom": "10px",
                            "text-align": "center"}),

            html.Div([html.H4("Male", style={'margin-bottom': '0'}),
                      html.H1(male_rate, style={'color': '#274D5A',
                                                'margin-top': '5px'})],
                     style={"border": "2px solid #eee",
                            "padding": "2px",
                            "margin-bottom": "10px",
                            "text-align": "center"}),

            html.Div([html.H4("Female", style={'margin-bottom': '0'}),
                      html.H1(female_rate, style={'color': '#447522',
                                                  'margin-top': '5px'})],
                     style={"border": "2px solid #eee",
                            "padding": "2px",
                            "margin-bottom": "10px",
                            "text-align": "center"}),
        ],
        style={"width": "100%", "margin-bottom": "10px"}),

        html.Div([
            dcc.Graph(figure=fig_p)
        ],
        style={"width": "100%", "text-align": "center"}),
    ],
    style={"float": "left", "padding": "5px"})
    ], width = 3),

    dbc.Col([
    html.Div([html.Div([
        html.H3("Australian Diabetes Map",
                style={'backgroundColor': '#6288BD',
                       'color': 'white',
                       'padding': '10px',
                       'text-align': 'center'}),
        dcc.Dropdown(
            id='map-selector',
            options=[
                {'label': 'Map by Postcode', 'value': 'postcode'},
                {'label': 'Map by State', 'value': 'state'}
            ],
            value='postcode',
            clearable=False,
            style={'width': '100%', 'margin-bottom': '20px'}
        ),
        html.Div(id='map-container', style={'width': '100%', 'overflow': 'hidden'})
    ])
    ])
    ],width=9),

    dbc.Row([
    html.Div([html.Div([
                    html.H2('Prevalence and Incidence',
                                style = {'backgroundColor': '#A1C67D',
                                          'color': 'white',
                                          'padding': '10px',
                                         'width': '100%'}),

             dbc.Row([
                    dbc.Col([
                        html.Div([dcc.Dropdown(id='measure-dropdown',
                                           options=[{'label': 'Prevalence', 'value': 'prevalence'},
                                                    {'label': 'Incidence', 'value': 'incidence'}],
                                           value='prevalence'
                                           )],
                             style={'display': 'inline-block', 'width': '30%'}),

                    html.Div([dcc.Dropdown(id='type-dropdown-1',
                                           options=[{'label': 'All', 'value': 'all'},
                                                    {'label': 'Type 1', 'value': 'type1'},
                                                    {'label': 'Type 2', 'value': 'type2'}],
                                           value='all'
                                           )],
                             style={'display': 'inline-block', 'width': '30%', 'margin-left': '20px'}),

                    html.Div([dcc.RadioItems(id='metric-radios',
                                             options=[],
                                             value='number',
                                             inline=True,
                                             labelStyle={'margin-right': '10px', 'padding-left': '10px'}
                                             )],
                             style={'margin-top': '5px'}),

                    dcc.Graph(id='time-series-graph', style={'margin-top': '2px', 'width': '100%'})
                ], width=6),

                dbc.Col([
                    html.Div([dcc.Dropdown(id='p-and-i-dropdown',
                                           options=[{'label': 'Prevalence', 'value': 'prevalence'},
                                                    {'label': 'Incidence', 'value': 'incidence'}],
                                           value='prevalence',
                                           style={'width': '50%', 'display': 'inline-block'}),
                              dcc.Dropdown(id='type-dropdown',
                                           options=[{'label': 'Type 1', 'value': 'type1'},
                                                    {'label': 'Type 2', 'value': 'type2'},
                                                    {'label': 'All', 'value': 'all'}],
                                           value='all',
                                           style={'width': '50%', 'display': 'inline-block'})
                              ], style={'margin-bottom': '10px'}),
                    dcc.Graph(id='age-graph', style={'width': '100%'})
                ], width=6)
            ]),

          dbc.Row([html.Div([html.Div([
              html.H2("Obesity And Type 2 Diabetes",
                                     style = {'backgroundColor' : '#6C818C',
                                              'color' : 'white',
                                              'padding' : '10px',
                                              'width': '100%'}),

              dbc.Row([
                  dbc.Col([
                      html.Div([dcc.Dropdown(id='age-group-selector',
                                            options=[{'label': 'Age 2-17', 'value': 'under_17'},
                                                      {'label': 'Age Above 18', 'value': 'above_18'}],
                                            value='under_17',
                                            clearable=False,
                                            style={'width': '50%', 'margin-bottom': '20px'}),
                                dcc.Graph(id='obesity-image', style={'width': '100%', 'text-align': 'center'})
                                ])
                  ], width=7),

                  dbc.Col([
                      html.Div([dcc.Graph(id='bmi-diabetes-bar-chart',
                                          figure=plt_E2,style={'width': '100%', 'margin-top': '60px'}),
                          ])
                  ], width=5)
              ])
          ])
    ])
])
])
])
])
])
])


@app.callback(
    dash.Output('map-container', 'children'),
    [dash.Input('map-selector', 'value')]
)
def update_map(selected_map):
    # Choose the map based on the selection
    if selected_map == 'state':
        map_html_content = state_map_content
    else:
        map_html_content = postcode_map_content

    # Display the selected map directly in a div
    return html.Iframe(
        srcDoc=map_html_content,
        width='100%',
        height='600'
    )

@app.callback(
    Output('metric-radios', 'options'),
    Input('measure-dropdown', 'value'),
    Input('type-dropdown-1', 'value'))
def set_radio_options(measure, type_):
    if not measure:
        measure = 'prevalence'
    if not type_:
        type_ = 'all'

    if measure == 'prevalence' and type_ == 'all':
        return [
            {'label': 'Number', 'value': 'number'},
            {'label': 'Percent', 'value': 'percent'}
        ]
    elif measure == 'prevalence' and type_ == 'type1':
        return [
            {'label': 'Number', 'value': 'number'},
            {'label': 'Number per 100,000 persons', 'value': 'per100k'}
        ]
    elif measure == 'prevalence' and type_ == 'type2':
        return [
            {'label': 'Number', 'value': 'number'},
            {'label': 'Percent', 'value': 'percent'}
        ]
    elif measure == 'incidence' and type_ == 'all':
        return [
            {'label': 'Number', 'value': 'number'},
            {'label': 'Number per 100,000 persons', 'value': 'per100k'}
        ]
    elif measure == 'incidence' and type_ == 'type1':
        return [
            {'label': 'Number', 'value': 'number'},
            {'label': 'Number per 100,000 persons', 'value': 'per100k'}
        ]
    elif measure == 'incidence' and type_ == 'type2':
        return [
            {'label': 'Number', 'value': 'number'},
            {'label': 'Number per 100,000 persons', 'value': 'per100k'}
        ]
    
@app.callback(
    Output('time-series-graph', 'figure'),
    Input('measure-dropdown', 'value'),
    Input('type-dropdown-1', 'value'),
    Input('metric-radios', 'value'))
def update_ts_graph(measure, type, options):
    # Prevalence
    if measure == 'prevalence':
        # Prevalence type1
        if type == 'type1':
            if options == 'number':
                fig1 = created_time_series_fig_number(type1_years_p)
                fig1.update_layout(yaxis_title = 'Number Count',
                                   xaxis = dict(nticks = 7, title = 'Year'),
                                   title = "Prevalence of Type 1 Diabetes(people under 19), 2014-2021",
                                   legend_title_text = 'Gender')
            else:
                fig1 = created_time_series_fig_100k(type1_years_p)
                fig1.update_layout(yaxis_title = 'Number per 100,000 persons',
                                   xaxis = dict(nticks = 7, title = 'Year'),
                                   title = "Prevalence of Type 1 Diabetes(people under 19), 2014-2021",
                                   legend_title_text = 'Gender')
        # Prevalence type 2
        elif type == 'type2':
            if options == 'number':
                fig1 = created_time_series_fig_number(type2_years_p)
                fig1.update_layout(yaxis_title = 'Number Count',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Prevalence of Type 2 Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
            else:
                fig1 = created_time_series_fig_percent(type2_years_p)
                fig1.update_layout(yaxis_title = 'Per cent',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Prevalence of Type 2 Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
        # Prevalence all
        else:
            if options == 'number':
                fig1 = created_time_series_fig_number(overall_year_p)
                fig1.update_layout(yaxis_title = 'Number Count',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Prevalence of All Type's Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
            else:
                fig1 = created_time_series_fig_percent(overall_year_p)
                fig1.update_layout(yaxis_title = 'Per cent',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Prevalence of All Type's Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')

    # Incidence
    else:
        # Incidence type1
        if type == 'type1':
            if options == 'number':
                fig1 = created_time_series_fig_number(type1_years_i)
                fig1.update_layout(yaxis_title = 'Number Count',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Incidence of Type 1 Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
            else:
                fig1 = created_time_series_fig_100k(type1_years_i)
                fig1.update_layout(yaxis_title = 'Number per 100,000 persons',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Incidence of Type 1 Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
        # Incidence type 2
        elif type == 'type2':
            if options == 'number':
                fig1 = created_time_series_fig_number(type2_years_i)
                fig1.update_layout(yaxis_title = 'Number Count',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Incidence of Type 2 Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
            else:
                fig1 = created_time_series_fig_100k(type2_years_i)
                fig1.update_layout(yaxis_title = 'Number per 100,000 persons',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Incidence of Type 2 Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
        # Incidence all
        else:
            if options == 'number':
                fig1 = created_time_series_fig_number(overall_year_i)
                fig1.update_layout(yaxis_title = 'Number Count',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Incidence of All Type's Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
            else:
                fig1 = created_time_series_fig_100k(overall_year_i)
                fig1.update_layout(yaxis_title = 'Number per 100,000 persons',
                                   xaxis = dict(nticks = 22, title = 'Year'),
                                   title = "Incidence of All Type's Diabetes, 2000-2021",
                                   legend_title_text = 'Gender')
    return fig1


@app.callback(
    Output('age-graph', 'figure'),
    Input('p-and-i-dropdown', 'value'),
    Input('type-dropdown', 'value'))
def update_graph_d2(select_pi, select_type):
    # Prevalence
    if select_pi == 'prevalence':
        if select_type == 'type1':
            fig2 = create_fig(type1_age_21p)
            fig2.update_layout(title = "Prevalence of Type 1 Diabetes's Population in 2021")
        elif select_type == 'type2':
            fig2 = create_fig(type2_age_21p)
            fig2.update_layout(title = "Prevalence of Type 2 Diabetes's Population in 2021")
        else:
            fig2 = create_fig(overall_age_21p)
            fig2.update_layout(title = "Prevalence of All Type Diabetes's Population in 2021")

    # Incidence
    else:
        if select_type == 'type1':
            fig2 = create_fig(type1_age_21i)
            fig2.update_layout(title = "Incidence of Type 1 Diabetes's Population in 2021")
        elif select_type == 'type2':
            fig2 = create_fig(type2_age_21i)
            fig2.update_layout(title = "Incidence of Type 2 Diabetes's Population in 2021")
        else:
            fig2 = create_fig(overall_age_21i)
            fig2.update_layout(title = "Incidence of All Type Diabetes's Population in 2021")

    return fig2


@app.callback(
    Output('obesity-image', 'figure'),
    [Input('age-group-selector', 'value')]
)
def update_image(selected_age_group):
    if selected_age_group == 'above_18':
        fig_E1 = px.bar(df_melted2,
             x='Age Group (Above 18)',
             y='Percentage',
             color='Category',
             barmode='group',
             title='Overweight and Obesity Rates for Adults (Age Above 18)',
             color_discrete_sequence=colors)

        fig_E1.update_layout(yaxis_title_text='Percentage (%)',
                             xaxis_title_text='Age Groups',
                             paper_bgcolor='white',
                             plot_bgcolor='white')
        fig_E1.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
        fig_E1.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
    else:
        fig_E1 = px.bar(df_melted1,
             x='Age Group (2-17)',
             y='Percentage',
             color='Category',
             barmode='group',
             title='Overweight and Obesity Rates for Children (Age 2-17)',
             color_discrete_sequence=colors)

        fig_E1.update_layout(yaxis_title_text='Percentage (%)',
                             xaxis_title_text='Age Groups',
                             paper_bgcolor='white',
                             plot_bgcolor='white')
        fig_E1.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
        fig_E1.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)

    return fig_E1

if __name__ == '__main__':
    app.run_server(debug=True)