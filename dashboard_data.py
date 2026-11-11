"""Data, provenance and rendering helpers for the earthquake dashboard."""
from pathlib import Path
from io import BytesIO
import os, json, html, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import folium
from folium.plugins import HeatMap
from branca.element import MacroElement, Template
from branca.colormap import LinearColormap
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parent
SOURCE='https://services.arcgis.com/eSGO1QW0lYo90eNU/arcgis/rest/services/ActiveFaultTrenches/FeatureServer/0'
COLORS=['#168C96','#DE9A26','#7556A8','#CD647F','#3259A2']
TECTONIC={'Active faults':'#b52637','Trenches':'#164c91','Collision zones':'#80469b'}
WINDOWS={'Whole catalog':None,'Northern window (Luzon)':(119,13,126,22),'Central window (Visayas)':(121,9,127,13),'Southern window (Mindanao)':(121,3,128,10)}

@st.cache_data(show_spinner=False)
def clean_catalog(data):
    d=pd.read_csv(BytesIO(data),low_memory=False)
    required=['id','time','updated','mag','depth','latitude','longitude','magType','tsunami']
    missing=set(required)-set(d.columns)
    if missing: raise ValueError('CSV is missing required columns: '+', '.join(sorted(missing)))
    audit={'source_rows':len(d)}
    for col in ['time','updated']:
        d[col]=pd.to_datetime(d[col],format='mixed',errors='coerce',utc=True)
    for col in ['mag','depth','latitude','longitude','tsunami']:
        d[col]=pd.to_numeric(d[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    if 'place' not in d:d['place']=''
    remove=[]
    for event_id,group in d.loc[d.id.notna() & d.id.duplicated(keep=False)].groupby('id',sort=False):
        if group.updated.isna().any():raise ValueError(f'Duplicate ID {event_id} needs review: missing update time.')
        latest=group.loc[group.updated.eq(group.updated.max())]
        if len(latest.drop_duplicates())>1:raise ValueError(f'Duplicate ID {event_id} has conflicting latest revisions.')
        remove.extend(group.index[group.index!=latest.index[0]])
    d=d.drop(index=remove).reset_index(drop=True)
    audit.update(removed_revisions=len(remove),cleaned_rows=len(d),missing_dates=int(d.time.isna().sum()))
    return d,audit

def find_catalog():
    override=os.environ.get('EARTHQUAKE_CSV')
    if override:return Path(override)
    local=ROOT/'data'/'Philippine_Earthquakes_USGS.csv'
    if local.exists():return local
    pinned=Path.home()/'.cache/kagglehub/datasets/bwandowando/philippine-earthquakes-1900-2025-from-usgs/versions/74/Philippine_Earthquakes_USGS.csv'
    return pinned if pinned.exists() else None

def geographic(d):
    return d.loc[d.latitude.between(-90,90)&d.longitude.between(-180,180)].copy()

def filter_catalog(d,years,magnitude,depth,types,include_missing=True,exclude_zero=False,window=None):
    mask=d.time.dt.year.between(*years)
    if include_missing:mask|=d.time.isna()
    for col,limits in [('mag',magnitude),('depth',depth)]:
        condition=d[col].between(*limits)
        if include_missing:condition|=d[col].isna()
        mask &= condition
    if types:mask &= d.magType.fillna('Unreported').isin(types)
    else:mask &= False
    if exclude_zero:mask &= d.mag.ne(0)
    if window:
        west,south,east,north=window
        mask &= d.longitude.between(west,east)&d.latitude.between(south,north)
    return d.loc[mask].copy()

@st.cache_data(show_spinner=False)
def grid_counts(d,size_km=50):
    d=geographic(d)
    if d.empty:return gpd.GeoDataFrame(columns=['events','center_lon','center_lat','geometry'],geometry='geometry',crs='EPSG:4326')
    p=gpd.GeoDataFrame(d,geometry=gpd.points_from_xy(d.longitude,d.latitude),crs=4326).to_crs(6933)
    size=size_km*1000
    counts=p.groupby([np.floor(p.geometry.x/size).astype(int),np.floor(p.geometry.y/size).astype(int)]).size()
    pairs=list(counts.index)
    g=gpd.GeoDataFrame({'events':counts.values},geometry=[box(x*size,y*size,(x+1)*size,(y+1)*size) for x,y in pairs],crs=6933)
    centers=g.geometry.centroid.to_crs(4326)
    g['center_lon']=centers.x;g['center_lat']=centers.y
    assert int(g.events.sum())==len(d)
    return g.to_crs(4326)

@st.cache_data(show_spinner=False)
def tectonics(geometry_bytes,source_bytes):
    try:
        source=json.loads(source_bytes); geo=json.loads(geometry_bytes)
        if source.get('source_url')!=SOURCE or source.get('publisher')!='aechanes_geodata':raise ValueError('Unverified provenance')
        if len(geo['features'])!=source['source_records']:raise ValueError('Incomplete cache record count')
        g=gpd.GeoDataFrame.from_features(geo['features'],crs=4326)
        missing=g.loc[g.geometry.isna(),'NAME'].fillna('Unnamed').tolist()
        g=g.loc[g.geometry.notna()].copy()
        if g.empty or not g.geom_type.isin(['LineString','MultiLineString']).all() or not g.is_valid.all():raise ValueError('Invalid line geometry')
        if not (g.total_bounds[0]>=110 and g.total_bounds[2]<=140 and g.total_bounds[1]>=0 and g.total_bounds[3]<=30):raise ValueError('Unexpected geographic extent')
        allowed={'Trace approximate','Active Fault','approximate offshore projection','Trench','Collision zone','Transform Fault'}
        if not set(g.DESC_).issubset(allowed):raise ValueError('Unrecognized feature classification')
        g['category']=np.where(g.DESC_.eq('Trench'),'Trenches',np.where(g.DESC_.eq('Collision zone'),'Collision zones','Active faults'))
        g['approximate']=g.DESC_.str.contains('approximate',case=False)
        return g,source,missing,None
    except (ValueError,KeyError,TypeError) as e:
        return None,{},[],str(e)

def load_tectonics():
    try:return tectonics((ROOT/'data/reference/arcgis_active_fault_trenches.geojson').read_bytes(),(ROOT/'data/reference/arcgis_active_fault_trenches.source.json').read_bytes())
    except OSError as e:return None,{},[],str(e)

def coverage(d):
    dates=d.time.dropna()
    return f'{dates.min():%d %b %Y} — {dates.max():%d %b %Y} UTC' if len(dates) else 'No dated records'

def polish(fig,height=380):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=15,r=20,t=35,b=20),font=dict(family='Arial',color='#253d4d',size=13),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',colorway=COLORS,legend_title_text='')
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor='#e8edf0',zeroline=False)
    return fig

class CanvasPoints(MacroElement):
    def __init__(self,rows,radius=3,opacity=.55):
        super().__init__();self._name='CanvasPoints';self.rows=json.dumps(rows,allow_nan=False);self.radius=radius;self.opacity=opacity
        self._template=Template('''{% macro script(this, kwargs) %}
        var canvas_{{this.get_name()}} = L.canvas({padding:0.5});
        {{this.rows}}.forEach(function(p){L.circleMarker([p[0],p[1]],{renderer:canvas_{{this.get_name()}},radius:{{this.radius}},color:p[2],weight:0,fillColor:p[2],fillOpacity:{{this.opacity}}}).bindTooltip(p[3]).addTo({{this._parent.get_name()}});});
        {% endmacro %}''')

@st.cache_data(show_spinner=False,max_entries=12)
def _build_map(d,mode='Heat',tectonic_visible=True,size_km=50,smoothing=16,color_field=None,view_bounds=None,sea_labels=False,point_color=None,point_radius=3,point_opacity=.55,tectonic_revision=''):
    d=geographic(d)
    m=folium.Map(location=[12.4,123.5],zoom_start=5,tiles=None,prefer_canvas=True,control_scale=True)
    folium.TileLayer('OpenStreetMap',name='OpenStreetMap',opacity=.55,control=False).add_to(m)
    geo,_,_,error=load_tectonics()
    folium.map.CustomPane('tectonic',z_index=650).add_to(m)
    if view_bounds is not None:
        west,south,east,north=view_bounds
        m.fit_bounds([[south,west],[north,east]])
    elif len(d):
        m.fit_bounds([[float(d.latitude.min())-.4,float(d.longitude.min())-.4],[float(d.latitude.max())+.4,float(d.longitude.max())+.4]])
    legend=None
    if mode=='Heat' and len(d):
        HeatMap(d[['latitude','longitude']].values.tolist(),name='Relative recorded-event concentration',radius=smoothing,blur=13,min_opacity=.15,gradient={.2:'#ffefa4',.45:'#f6bd54',.7:'#eb793d',1.:'#ab263c'}).add_to(m)
    elif mode=='Count grid' and len(d):
        grid=grid_counts(d,size_km)
        max_count=int(grid.events.max())
        legend=LinearColormap(['#fff0ae','#ef9e48','#ad273b'],vmin=0,vmax=float(np.log1p(max_count)),caption=f'log(1 + recorded events) per {size_km**2:,} km² cell; hover for exact count')
        folium.GeoJson(json.loads(grid.to_json()),name='Exact cell counts',style_function=lambda f:{'fillColor':legend(np.log1p(f['properties']['events'])),'fillOpacity':.65,'color':'#ffffff','weight':.2},tooltip=folium.GeoJsonTooltip(fields=['events','center_lat','center_lon'],aliases=['Recorded events','Cell latitude','Cell longitude'],localize=True)).add_to(m)
    elif len(d) and mode!='Reference only':
        if color_field in ['profile','cluster']:
            colors=d[color_field].map(lambda x:'#9aa4aa' if x==-1 else (COLORS[(int(x)-1)%len(COLORS)] if color_field=='profile' else px.colors.qualitative.Dark24[int(x)%24])).tolist()
        elif color_field in ['mag','depth']:
            values=d[color_field];finite=values.dropna()
            if len(finite):
                legend=LinearColormap(['#169999','#e3b743','#b52843'],vmin=float(finite.min()),vmax=max(float(finite.max()),float(finite.min())+.001),caption='Magnitude' if color_field=='mag' else 'Depth (km)')
                colors=[legend(v) if pd.notna(v) else '#999999' for v in values]
            else:colors=['#999999']*len(d)
        else:colors=[point_color or COLORS[0]]*len(d)
        group=folium.FeatureGroup(name='Recorded epicenters').add_to(m)
        rows=[]
        for (_,row),color in zip(d.iterrows(),colors):
            text=html.escape(f"{row.get('id','')} · M {row.mag:g} · {row.depth:g} km · {row.get('place','')}")
            if color_field in ['profile','cluster']:
                text += (' · Profile ' if color_field=='profile' else ' · Group ') + str(int(row[color_field]))
            # Escape HTML delimiters before embedding a tooltip string in a script.
            text=text.replace('<','&lt;').replace('>','&gt;')
            rows.append([float(row.latitude),float(row.longitude),color,text])
        CanvasPoints(rows,point_radius,point_opacity).add_to(group)
    if legend:legend.add_to(m)
    if geo is not None:
        for category,layer in geo.groupby('category'):
            group=folium.FeatureGroup(name=category+' · provisional',show=tectonic_visible).add_to(m)
            folium.GeoJson(json.loads(layer.to_json()),pane='tectonic',control=False,style_function=lambda f:{'color':'white','weight':5,'opacity':.95}).add_to(group)
            folium.GeoJson(json.loads(layer.to_json()),pane='tectonic',control=False,style_function=lambda f,category=category:{'color':TECTONIC[category],'weight':2,'opacity':1,'dashArray':'7 4' if f['properties']['approximate'] or category=='Trenches' else None},tooltip=folium.GeoJsonTooltip(fields=['NAME','DESC_'])).add_to(group)
    if error:
        m.get_root().html.add_child(folium.Element('<div style="background:#fff0f0;color:#b52637;padding:12px;font-weight:bold">TECTONIC OVERLAY UNAVAILABLE</div>'))
    if sea_labels:
        for label,lon,lat in [('South China Sea',117.8,17.5),('Philippine Sea',130.,17.),('Sulu Sea',120.3,8.3),('Celebes Sea',123.,3.5)]:
            folium.Marker([lat,lon],icon=folium.DivIcon(html=f'<span style="color:#174c6d;font:italic 12px Arial;white-space:nowrap;text-shadow:0 0 3px white">{label}</span>',icon_size=(110,18),icon_anchor=(55,9))).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()


def build_map(*args, **kwargs):
    """Invalidate rendered maps when tectonic assets arrive or change on deployment."""
    revision=[]
    for filename in ['arcgis_active_fault_trenches.geojson','arcgis_active_fault_trenches.source.json']:
        try:
            revision.append(hashlib.sha256((ROOT/'data/reference'/filename).read_bytes()).hexdigest())
        except OSError:
            revision.append('unavailable')
    return _build_map(*args, **kwargs, tectonic_revision=tuple(revision))
