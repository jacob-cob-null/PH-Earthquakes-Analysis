"""Run: streamlit run streamlit_app.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from dashboard_data import (ROOT,COLORS,TECTONIC,WINDOWS,SOURCE,clean_catalog,find_catalog,
    geographic,filter_catalog,grid_counts,load_tectonics,coverage,polish,build_map)
from dashboard_models import prepare_profiles,fit_profiles,profile_features

st.set_page_config(page_title='Philippine Earthquakes',page_icon='🌏',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background:#f7f9fa} .block-container{padding-top:1rem;padding-bottom:3rem;max-width:1600px}
[data-testid="stSidebar"]{background:#edf2f4} h1,h2,h3{letter-spacing:-.035em;color:#123346}
[data-testid="stMetric"]{background:white;border:1px solid #e0e8ec;border-radius:12px;padding:10px 14px}
[data-testid="stMetricLabel"]{color:#617685} [data-testid="stMetricValue"]{font-size:1.6rem;color:#173c50}
[data-testid="stPlotlyChart"]{background:white;border:1px solid #e1e8eb;border-radius:12px;padding:5px}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_profiles(d):
    subset,audit,kind=prepare_profiles(d)
    # Reuse notebook assignments only when the complete training subset matches.
    path=ROOT/'outputs/ml_profile_memberships.csv'
    if path.exists() and not subset.id.isna().any() and subset.id.is_unique:
        saved=pd.read_csv(path,low_memory=False)
        if saved.id.is_unique and len(saved)==len(subset) and set(saved.id)==set(subset.id):
            a=subset.set_index('id').sort_index();b=saved.set_index('id').sort_index()
            if all(np.allclose(a[c],b[c],equal_nan=True) for c in ['mag','depth','latitude','longitude']) and a.magType.equals(b.magType):
                profile_cols=['id','profile','membership_certainty']+[c for c in saved if c.startswith('probability_profile_')]
                subset=subset.merge(saved[profile_cols],on='id',validate='one_to_one')
                selection=pd.read_csv(ROOT/'outputs/ml_profile_model_selection.csv') if (ROOT/'outputs/ml_profile_model_selection.csv').exists() else pd.DataFrame()
                robust=pd.read_csv(ROOT/'outputs/ml_profile_robustness.csv') if (ROOT/'outputs/ml_profile_robustness.csv').exists() else pd.DataFrame()
                return subset,audit,kind,selection,robust,'Notebook model (matching training records)'
    selection,model,_=fit_profiles(subset,'Primary: standardized magnitude + log1p depth')
    if model is None:return subset.iloc[:0],audit,kind,selection,pd.DataFrame(),'No converged profile model'
    X,_=profile_features(subset);labels=model.predict(X);prob=model.predict_proba(X)
    order=sorted(range(model.n_components),key=lambda j:(subset.loc[labels==j,'depth'].median() if (labels==j).any() else np.inf,j))
    mapping={raw:i+1 for i,raw in enumerate(order)}
    subset['profile']=[mapping[j] for j in labels];subset['membership_certainty']=prob.max(axis=1)
    for raw in order:subset[f'probability_profile_{mapping[raw]}']=prob[:,raw]
    return subset,audit,kind,selection,pd.DataFrame(),'Fitted on the full eligible catalog'

@st.cache_data(show_spinner=False)
def cluster_lab(d,radius,minimum):
    geo=geographic(d)
    labels=DBSCAN(eps=radius/6371.0088,min_samples=minimum,metric='haversine',algorithm='ball_tree').fit_predict(np.radians(geo[['latitude','longitude']]))
    geo['cluster']=labels
    return geo

@st.cache_data(show_spinner=False)
def cluster_sensitivity(d):
    rows=[]
    for radius in [15,25,40]:
        for minimum in [10,20,40]:
            result=cluster_lab(d,radius,minimum);sizes=result.loc[result.cluster.ne(-1),'cluster'].value_counts()
            rows.append(dict(radius_km=radius,min_samples=minimum,clusters=len(sizes),largest_share=sizes.max()/len(result) if len(sizes) else 0,sparse_share=result.cluster.eq(-1).mean()))
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def unusual(d):
    subset=d.loc[d.mag.gt(0)&d.depth.ge(0)].copy();types=subset.magType.value_counts()
    if types.empty:return subset.iloc[:0],None
    kind=types.index[0];subset=subset.loc[subset.magType.eq(kind)].copy()
    if len(subset)<20 or subset[['mag','depth']].nunique().le(1).all():return subset.iloc[:0],kind
    with threadpool_limits(limits=1):model=IsolationForest(n_estimators=100,random_state=42).fit(subset[['depth','mag']])
    subset['unusual_score']=-model.score_samples(subset[['depth','mag']]);return subset,kind

def chart(fig,key,height=380,selection=False):
    return st.plotly_chart(polish(fig,height),width='stretch',key=key,on_select='rerun' if selection else 'ignore',selection_mode=('box','lasso'),config={'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})

def map_source_note():
    g,source,missing,error=load_tectonics()
    if error:st.error('Tectonic overlay unavailable — '+error)
    else:
        st.caption('Faults: red · trenches: blue dashed · collision zones: purple · dashed fault traces: approximate.')
        st.caption(f"Provisional layer: {source['publisher']} · Missing geometry: {', '.join(missing) or 'none'} · © OpenStreetMap contributors.")
        with st.expander('Map source details'):
            st.write('Original authority and mapping date unspecified. Geometry is incomplete. Proximity does not identify the source fault.')
            st.link_button('Tectonic source',SOURCE)
            st.caption('Map tiles and JavaScript require internet.')

def map_view(d,mode='Heat',field=None,key='map',smoothing=16,size=50,view_bounds=None,sea_labels=False,point_color=None,notes=True,height=580,point_radius=3,point_opacity=.55):
    if geographic(d).empty and view_bounds is None:st.info('No valid coordinates in this selection.');return
    markup=build_map(d,mode,size_km=size,smoothing=smoothing,color_field=field,view_bounds=view_bounds,sea_labels=sea_labels,point_color=point_color,point_radius=point_radius,point_opacity=point_opacity)
    st.iframe(markup,height=height,alt="Earthquake map with tectonic layer controls")
    st.download_button('Download map',markup,file_name=f'{key}.html',mime='text/html',key='download_'+key)
    if notes:map_source_note()


def table_export(d,name):
    st.dataframe(d,width='stretch',hide_index=True)
    st.download_button('Download table (CSV)',d.to_csv(index=False).encode('utf-8'),file_name=name+'.csv',mime='text/csv',key='table_download_'+name)

with st.sidebar:
    st.markdown('### Philippine earthquakes')
    st.caption('PHILIPPINES · HISTORICAL CATALOG')
    with st.expander('Data source',expanded=False):
        uploaded=st.file_uploader('Use a USGS-format catalog CSV',type='csv')
        located=find_catalog()
        if located:st.caption('Local source: '+str(located))
        if st.button('Download pinned Kaggle version 74'):
            import kagglehub
            with st.spinner('Downloading catalog version 74…'):
                try:
                    folder=Path(kagglehub.dataset_download('bwandowando/philippine-earthquakes-1900-2025-from-usgs/versions/74'))
                    st.session_state['downloaded_catalog']=str(next(folder.rglob('*.csv')))
                except Exception as error:
                    st.error('Catalog download failed. You can upload the CSV instead. '+str(error))
    source=uploaded.getvalue() if uploaded else None
    located=Path(st.session_state['downloaded_catalog']) if 'downloaded_catalog' in st.session_state else located
    if source is None and located:
        try:source=located.read_bytes()
        except OSError as e:st.error(str(e))
    if source is None:
        st.info('Upload a CSV or download the pinned catalog to begin.');st.stop()
    try:catalog,audit=clean_catalog(source)
    except (ValueError,OSError,pd.errors.ParserError) as e:st.error(str(e));st.stop()
    if catalog.empty or catalog.time.notna().sum()==0:st.error('A nonempty catalog with dated events is required.');st.stop()
    st.markdown('#### Filters')
    y0,y1=int(catalog.time.dt.year.min()),int(catalog.time.dt.year.max())
    years=st.slider('UTC years',y0,y1,(y0,y1)) if y0<y1 else (y0,y1)
    def numeric_range(col,label):
        vals=catalog[col].dropna()
        if vals.empty:return (-1e6,1e6)
        lo,hi=float(vals.min()),float(vals.max())
        return st.slider(label,lo,hi,(lo,hi),step=.1) if lo<hi else (lo,hi)
    magnitude=numeric_range('mag','Magnitude range')
    depth=numeric_range('depth','Depth range (km)')
    window_name=st.selectbox('Geographic window',list(WINDOWS))
    st.caption('Named windows are map rectangles, not administrative boundaries.')
    with st.expander('Catalog options'):
        include_missing=st.checkbox('Include records missing date / magnitude / depth',value=True)
        exclude_zero=st.checkbox('Exclude unresolved zero magnitudes',value=False)
        all_types=sorted(catalog.magType.fillna('Unreported').astype(str).unique())
        types=st.multiselect('Reported magnitude scales',all_types,default=all_types)
    filtered=filter_catalog(catalog,years,magnitude,depth,types,include_missing,exclude_zero,WINDOWS[window_name])
    st.caption(f'{len(filtered):,} / {len(catalog):,} cleaned records selected')
    st.download_button('Export selected records',filtered.to_csv(index=False),file_name='earthquake_selection.csv',mime='text/csv')

st.title('Philippine Earthquakes')
section=st.radio('View',['Q1 · Geography','Q2 · Frequency','Q3 · Magnitude & depth','Q4 · Tsunami flags','ML · Profiles','Appendix'],horizontal=True,label_visibility='collapsed')
metrics=st.columns(4)
metrics[0].metric('Selected records',f'{len(filtered):,}')
metrics[1].metric('Mapped epicenters',f'{len(geographic(filtered)):,}')
metrics[2].metric('Magnitude ≥ 6',f'{filtered.mag.ge(6).sum():,}')
metrics[3].metric('Median depth',f'{filtered.depth.median():.1f} km' if filtered.depth.notna().any() else 'Unavailable')
st.caption(coverage(filtered)+' · Historical catalog coverage')
if filtered.empty:st.info('No records match these filters. Widen the ranges or restore magnitude scales in Catalog options.');st.stop()

if section=='Q1 · Geography':
    layout=st.radio('Geographic figures',['Concentration','Magnitude & depth maps','Northern / southern close-ups'],horizontal=True)
    valid=geographic(filtered)
    shared_bounds=(float(valid.longitude.min())-.4,float(valid.latitude.min())-.4,float(valid.longitude.max())+.4,float(valid.latitude.max())+.4) if len(valid) else None
    if layout=='Magnitude & depth maps':
        for col,field,title in zip(st.columns(2),['mag','depth'],['Magnitude','Depth (km)']):
            with col:
                st.markdown('#### '+title)
                map_view(filtered,'Epicenters',field=field,key='overview_'+field,view_bounds=shared_bounds,notes=False,height=640)
        map_source_note()
    elif layout=='Northern / southern close-ups':
        for col,title,view in zip(st.columns(2),['Northern close-up','Southern close-up'],[(119,13,126,22),(121,3,128,12)]):
            with col:
                st.markdown('#### '+title)
                map_view(filtered,'Epicenters',field='mag',key=title.split()[0].lower(),view_bounds=view,notes=False,height=640)
        map_source_note()
    else:
        a,b,c=st.columns([2,1,1])
        mode=a.selectbox('Map view',['Heat','Epicenters','Count grid','Reference only'])
        color=b.selectbox('Color epicenters by',['Magnitude','Depth'])
        smoothing=c.slider('Heat smoothing',8,32,16) if mode=='Heat' else 16
        size=c.selectbox('Equal-area cell side (km)',[25,50,100],index=1) if mode=='Count grid' else 50
        map_view(filtered,mode,field='mag' if color=='Magnitude' else 'depth',smoothing=smoothing,size=size,key='catalog_map')
        st.caption('Heat: relative concentration, equal event weights. Grid: exact recorded-event counts. Neither measures hazard or damage risk.')
        grid=grid_counts(filtered,size)
        if len(grid):
            top=grid.nlargest(10,'events').sort_values('events')
            top['Cell center']=top.apply(lambda r:f'{r.center_lat:.2f}°N, {r.center_lon:.2f}°E',axis=1)
            chart(px.bar(top,x='events',y='Cell center',orientation='h',color_discrete_sequence=[COLORS[0]],labels={'events':'Recorded events'},title=f'Top 10 equal-area cells · {size**2:,} km² per cell'),'topcells',380)
            with st.expander('Exact geographic counts'):
                st.caption(f'{int(grid.events.sum()):,} events · {len(grid):,} occupied cells')
                table_export(grid.drop(columns='geometry'),'geographic_counts')

elif section=='Q2 · Frequency':
    aggregation=st.radio('Count interval',['Year','Month'],horizontal=True)
    dated=filtered.dropna(subset=['time']).copy()
    if dated.empty:st.info('No dated events match the selection.')
    else:
        freq='YS' if aggregation=='Year' else 'MS'
        period='Y' if aggregation=='Year' else 'M'
        index=pd.date_range(dated.time.min().tz_localize(None).to_period(period).start_time,dated.time.max().tz_localize(None).to_period(period).start_time,freq=freq,tz='UTC')
        series=pd.DataFrame(index=index)
        for threshold in [6,7]:
            series[f'M ≥ {threshold}']=dated.loc[dated.mag.ge(threshold)].set_index('time').resample(freq).size().reindex(index,fill_value=0)
            assert series[f'M ≥ {threshold}'].sum()==dated.mag.ge(threshold).sum()
        long=series.rename_axis('UTC period').reset_index().melt(id_vars='UTC period',var_name='Magnitude',value_name='Recorded events')
        fig=px.line(long,x='UTC period',y='Recorded events',color='Magnitude',color_discrete_sequence=[COLORS[0],COLORS[3]],title='Recorded earthquakes · M ≥ 6 and M ≥ 7')
        step=pd.DateOffset(years=1) if aggregation=='Year' else pd.DateOffset(months=1)
        for boundary in sorted(set([index[0],index[-1]])):
            fig.add_vrect(x0=boundary,x1=boundary+step,fillcolor='#7d8c97',opacity=.15,line_width=0,layer='below')
        fig.update_layout(hovermode='x unified');chart(fig,'timeline',440)
        st.caption('Shaded boundary periods may be partial. Catalog coverage and magnitude methods vary over time.')
        annual=dated.groupby(dated.time.dt.year).size().rename('events').reset_index(name='events')
        annual['decade']=(annual.time//10)*10;annual['year_in_decade']=annual.time%10
        chart(px.density_heatmap(annual,x='year_in_decade',y='decade',z='events',histfunc='sum',color_continuous_scale=['#edf5f5','#168c96','#123346'],labels={'year_in_decade':'Year within decade','decade':'Decade','events':'Recorded events'},title='All recorded events by year'),'calendar',330)
        with st.expander('Period counts'):table_export(series.rename_axis('UTC period').reset_index(),'period_counts')

elif section=='Q3 · Magnitude & depth':
    left,right=st.columns(2)
    with left:
        values=filtered.mag.dropna()
        fig=go.Figure()
        if len(values):
            edges=np.histogram_bin_edges(values,bins=45)
            bins=dict(start=float(edges[0]),end=float(edges[-1]),size=float(edges[1]-edges[0]))
            fig.add_histogram(x=values,name='All available',xbins=bins,marker_color='#a6b7bf',opacity=.7)
            fig.add_histogram(x=values.loc[values.ne(0)],name='Excluding zero',xbins=bins,marker=dict(color='rgba(22,140,150,.12)',line=dict(color=COLORS[0],width=1.5)))
        fig.update_layout(barmode='overlay',title='Magnitude distribution',xaxis_title='Magnitude',yaxis_title='Recorded events',legend=dict(orientation='h',y=-.22,x=0,xanchor='left',yanchor='top'))
        chart(fig,'mag_hist')
    with right:chart(px.histogram(filtered,x='depth',nbins=50,color_discrete_sequence=[COLORS[2]],labels={'depth':'Depth (km)'},title='Depth distribution'),'depth_hist')
    pairs=filtered.dropna(subset=['mag','depth']).copy()
    if len(pairs):
        pairs['Scale']=pairs.magType.fillna('Unreported')
        fig=px.scatter(pairs,x='depth',y='mag',color='Scale',hover_name='place',hover_data=['id','time'],custom_data=['id'],opacity=.45,render_mode='webgl',labels={'depth':'Depth (km)','mag':'Magnitude'},title='Magnitude and depth')
        event=chart(fig,'depth_mag',440,selection=True)
        selected=[p['customdata'][0] for p in event.selection.points if 'customdata' in p]
        if selected:table_export(pairs.loc[pairs.id.isin(selected),['id','time','place','mag','magType','depth']],'selected_points')
    categories=pd.cut(filtered.loc[filtered.depth.ge(0),'depth'],[0,70,300,np.inf],right=False,labels=['0–<70 km','70–<300 km','≥300 km'])
    counts=categories.value_counts(sort=False).rename_axis('Depth category').reset_index(name='Recorded events')
    a,b=st.columns(2)
    with a:chart(px.pie(counts,names='Depth category',values='Recorded events',hole=.65,color_discrete_sequence=COLORS,title='Depth categories · all nonnegative depths'),'depth_categories',360)
    with b:
        repeated=filtered.depth.value_counts().head(12).rename_axis('Depth (km)').reset_index(name='Recorded events')
        repeated['Depth (km)']=repeated['Depth (km)'].map(lambda x:f'{x:g} km')
        chart(px.bar(repeated.iloc[::-1],x='Recorded events',y='Depth (km)',orientation='h',color_discrete_sequence=[COLORS[2]],title='Most frequently reported depths'),'repeated_depths',360)
    if len(pairs):
        comparisons={'All paired records':pairs,'Excluding zero magnitude':pairs.loc[pairs.mag.ne(0)]}
        scales=pairs.magType.value_counts()
        if len(scales):comparisons[f'Most common scale: {scales.index[0]}']=pairs.loc[pairs.magType.eq(scales.index[0])]
        correlations=[]
        for name,group in comparisons.items():
            rho=group[['depth','mag']].rank().corr().loc['depth','mag'] if len(group)>=3 else np.nan
            correlations.append({'Subset':name,'Records':len(group),'Spearman ρ':rho})
        st.dataframe(pd.DataFrame(correlations).round(4),hide_index=True,width='stretch')
    st.caption(f'Unresolved zero magnitudes: {filtered.mag.eq(0).sum():,} · Exact 10 km depths: {filtered.depth.eq(10).sum():,} · Exact 33 km depths: {filtered.depth.eq(33).sum():,}. Records are retained unless excluded by the filters.')
    with st.expander('Descriptive statistics'):st.dataframe(filtered[['mag','depth']].describe().round(2),width='stretch')

elif section=='Q4 · Tsunami flags':
    valid=geographic(filtered)
    flagged=valid.loc[valid.tsunami.eq(1)].copy()
    if valid.empty:st.info('No valid coordinates in this selection.')
    else:
        view=(float(valid.longitude.min())-.4,float(valid.latitude.min())-.4,float(valid.longitude.max())+.4,float(valid.latitude.max())+.4)
        for column,frame,title,color in zip(st.columns(2),[valid,flagged],['All mapped events','Tsunami flag = 1'],[COLORS[0],'#d85e00']):
            with column:
                st.markdown(f'#### {title} · {len(frame):,}')
                st.caption(coverage(frame))
                map_view(frame,'Epicenters',key='tsunami_all' if title=='All mapped events' else 'tsunami_flagged',view_bounds=view,sea_labels=True,point_color=color,notes=False,height=600,point_radius=4,point_opacity=.8)
        st.caption('Same map extent and marker size. Flag = 1 does not confirm a tsunami; oceanic location is part of the flag definition. Missing earlier flags do not establish absence of historical tsunamis.')
        map_source_note()
    if len(flagged):
        annual=flagged.dropna(subset=['time']).groupby(flagged.time.dt.year).size().rename_axis('UTC year').reset_index(name='Flagged records')
        chart(px.bar(annual,x='UTC year',y='Flagged records',color_discrete_sequence=[COLORS[3]],title='Tsunami-flagged records by year'),'flag_years',280)
        with st.expander('Flagged records'):table_export(flagged[['id','time','place','mag','depth','tsunami']],'flagged_records')

elif section=='ML · Profiles':
    st.subheader('Magnitude–depth profiles')
    with st.spinner('Loading magnitude–depth profiles…'):profiles,paudit,kind,selection,robust,model_source=get_profiles(catalog)
    if profiles.empty:st.info('No eligible converged profile model.');st.dataframe(paudit);st.stop()
    st.caption(f'{model_source} · trained on {len(profiles):,} type={kind} records · {coverage(profiles)}. Filters change the displayed records; the full-catalog model stays fixed.')
    current=profiles.loc[profiles.id.isin(filtered.id)].copy()
    options=sorted(profiles.profile.unique());chosen=st.multiselect('Show profiles',options,default=options,format_func=lambda x:f'Profile {x}')
    current=current.loc[current.profile.isin(chosen)]
    if current.empty:st.info('No profile records match the filters. Restore profiles or widen the selection.');st.stop()
    current['Profile']=current.profile.map(lambda x:f'Profile {x}')
    color_map={f'Profile {i}':COLORS[(i-1)%len(COLORS)] for i in options}
    a,b=st.columns([2,1])
    with a:
        fig=px.scatter(current,x='depth',y='mag',color='Profile',color_discrete_map=color_map,category_orders={'Profile':list(color_map)},hover_data=['id','place','membership_certainty'],custom_data=['id'],opacity=.45,render_mode='webgl',labels={'depth':'Depth (km)','mag':f'Magnitude ({kind})'},title='Magnitude–depth profiles')
        picked=chart(fig,'profiles_scatter',450,True)
    with b:
        summary=current.groupby('Profile').agg(records=('id','size'),median_depth_km=('depth','median'),median_magnitude=('mag','median'),median_certainty=('membership_certainty','median')).reset_index()
        chart(px.bar(summary,x='records',y='Profile',color='Profile',color_discrete_map=color_map,orientation='h',title='Profiles in this selection'),'profile_sizes',450)
    selected_ids=[p['customdata'][0] for p in picked.selection.points if 'customdata' in p]
    map_records=current.loc[current.id.isin(selected_ids)] if selected_ids else current
    st.markdown('#### Profile locations')
    st.caption(f'{len(map_records):,} mapped profile records. Box/lasso selection above also focuses this map. Clear the selection to restore all visible profiles.')
    map_view(map_records,'Epicenters',field='profile',key='profile_map')
    st.markdown('#### Depth categories and model sensitivity')
    st.caption('Profile colors describe statistical groups. High membership certainty describes the fitted model; it is not an earthquake or hazard probability.')
    current['Depth category']=pd.cut(current.depth,[0,70,300,np.inf],right=False,labels=['0–<70 km','70–<300 km','≥300 km'])
    cross=pd.crosstab(current['Depth category'],current.Profile).reindex(['0–<70 km','70–<300 km','≥300 km'],fill_value=0)
    shares=cross.div(cross.sum(axis=1).replace(0,np.nan),axis=0).fillna(0).mul(100).reset_index().melt(id_vars='Depth category',value_name='Share (%)',var_name='Profile')
    a,b=st.columns(2)
    with a:chart(px.bar(shares,x='Share (%)',y='Depth category',color='Profile',orientation='h',color_discrete_map=color_map,title='Profile composition by depth category'),'profile_depth')
    with b:
        if len(robust):
            checks=robust.loc[robust.valid.eq(True)].copy();checks['Check']=checks['check'].str.replace('Primary preprocessing; ','',regex=False).str.replace('Sensitivity: ','',regex=False)
            chart(px.bar(checks,x='agreement_ARI',y='Check',orientation='h',color_discrete_sequence=['#607d8b'],range_x=[0,1.05],labels={'agreement_ARI':'Agreement with primary grouping (ARI)'},title='Full training-subset robustness'),'robust',380)
        else:st.info('Full robustness checks are available by executing the notebook ML extension. No robustness claim is made for this newly fitted catalog.')
    st.caption('The tested range is 1–5 components. A selection at five is a boundary result, not evidence of five natural earthquake types. BIC is comparable within one preprocessing variant.')
    with st.expander('Profile summaries, model selection and training audit'):
        table_export(summary,'profile_summary');st.dataframe(selection,width='stretch');st.dataframe(paudit,width='stretch')
    st.download_button('Download displayed profile memberships',current.to_csv(index=False),file_name='profile_memberships.csv')

else:
    st.subheader('Supplementary analysis')
    experiment=st.radio('Inspect',['Spatial clustering','Unusual records','Catalog quality & sources'],horizontal=True)
    if experiment=='Spatial clustering':
        a,b=st.columns(2);radius=a.select_slider('Neighborhood radius (km)',[15,25,40],value=25);minimum=b.select_slider('Minimum records',[10,20,40],value=20)
        if geographic(filtered).empty:st.info('No valid coordinates for clustering.');st.stop()
        with st.spinner('Grouping selected coordinates…'):result=cluster_lab(filtered,radius,minimum)
        sizes=result.loc[result.cluster.ne(-1)].cluster.value_counts()
        a,b,c=st.columns(3);a.metric('Groups',len(sizes));b.metric('Largest group share',f'{sizes.max()/len(result):.1%}' if len(sizes) else '0%');c.metric('Sparse-event share',f'{result.cluster.eq(-1).mean():.1%}')
        st.caption('This experiment refits the filtered selection. Groups connect dense neighborhoods and may span much farther than the radius. Sparse events are not errors. Map colors repeat after 24 groups; use group IDs in the table.')
        map_view(result,'Epicenters',field='cluster',key='cluster_map')
        if st.button('Compare all nine settings'):
            with st.spinner('Comparing settings…'):sensitivity=cluster_sensitivity(filtered)
            matrix=sensitivity.pivot(index='min_samples',columns='radius_km',values='largest_share')
            chart(px.imshow(matrix,text_auto='.0%',color_continuous_scale=['#ecf4f4','#168c96','#123346'],labels={'x':'Radius (km)','y':'Minimum records','color':'Largest group share'},title='Largest-cluster share across settings'),'cluster_sensitivity',350)
            table_export(sensitivity,'cluster_sensitivity')
        with st.expander('Group membership'):table_export(result[['id','time','mag','depth','cluster']],'cluster_memberships')
    elif experiment=='Unusual records':
        with st.spinner('Ranking selected records…'):ranked,kind=unusual(filtered)
        if ranked.empty:st.info('At least 20 eligible, varying records on one magnitude scale are needed.');st.stop()
        ranked=ranked.sort_values('unusual_score',ascending=False);top=ranked.head(10)
        st.caption(f'Type={kind}; positive magnitudes and nonnegative depths. Isolation Forest ranks unusual combinations within this filtered subset. A high score does not establish an error. All records are retained.')
        fig=px.scatter(ranked,x='depth',y='mag',color='unusual_score',hover_data=['id','place'],color_continuous_scale=['#d5e8e9','#168c96','#bc3a4b'],opacity=.6,render_mode='webgl',labels={'depth':'Depth (km)','mag':'Magnitude','unusual_score':'Unusualness'})
        fig.add_scatter(x=top.depth,y=top.mag,mode='markers',marker=dict(size=14,color='rgba(0,0,0,0)',line=dict(color='#bc3a4b',width=2)),name='Ten highest review scores')
        chart(fig,'unusual',480);table_export(top[['id','time','place','mag','depth','magType','unusual_score']],'unusual_review')
    else:
        a,b=st.columns([1,1.3])
        with a:
            st.markdown('#### Cleaning audit');st.json(audit)
            missing=filtered[['time','mag','depth','latitude','longitude','magType']].isna().sum().rename_axis('Field').reset_index(name='Missing records')
            chart(px.bar(missing,x='Missing records',y='Field',orientation='h',color_discrete_sequence=[COLORS[0]],title='Missing values in this selection'),'missing',330)
        with b:
            st.markdown('#### Methods and sources')
            st.write('Same-ID revisions retain the latest update; conflicting revisions require review. Unusual magnitudes and depths are retained. Felt-report analysis remains deferred. Date filters use UTC. Geographic concentration describes recorded locations, and catalog completeness varies through time.')
            st.write('The count grid uses EPSG:6933 equal-area cells. Heat assigns equal weight to every valid-coordinate event. GMM uses standardized magnitude and log-transformed depth on one magnitude scale, with a 1–5 component comparison. DBSCAN uses haversine distance. Isolation Forest supplies review rankings.')
            geo,provenance,missing_lines,error=load_tectonics()
            if error:st.error('Tectonic overlay unavailable: '+error)
            else:
                st.write(f"**Tectonic publisher:** {provenance['publisher']} · **{len(geo)}** usable line features / **{provenance['source_records']}** source records.")
                st.write('Original authority: unspecified. Mapping date: unspecified. Retrieved: '+provenance.get('retrieved_utc','unspecified'))
                st.warning('Incomplete provisional reference. Missing geometry: '+', '.join(missing_lines))
                st.link_button('Inspect tectonic source',SOURCE)
            st.link_button('Dataset on Kaggle','https://www.kaggle.com/datasets/bwandowando/philippine-earthquakes-1900-2025-from-usgs')
        with st.expander('Magnitude scales & potential outliers'):
            scale_counts=filtered.magType.fillna('Unreported').value_counts().rename_axis('Magnitude scale').reset_index(name='Records')
            chart(px.bar(scale_counts,x='Magnitude scale',y='Records',color_discrete_sequence=[COLORS[0]],title='Reported magnitude scales'),'scales',300)
            left,right=st.columns(2)
            audit_rows=[]
            for column,target in [('mag',left),('depth',right)]:
                values=filtered[column].dropna();q1,q3=values.quantile([.25,.75]) if len(values) else (np.nan,np.nan)
                iqr=q3-q1;lo,hi=q1-1.5*iqr,q3+1.5*iqr
                audit_rows.append({'Field':column,'IQR lower':lo,'IQR upper':hi,'Potential outliers':int(((values<lo)|(values>hi)).sum())})
                with target:chart(px.box(filtered,x=column,points='outliers',color_discrete_sequence=[COLORS[0]],title=column+' · potential outliers'),'box_'+column,250)
            st.dataframe(pd.DataFrame(audit_rows),hide_index=True)
            st.caption('IQR flags are descriptive review aids. Records are retained, including physically plausible extremes.')
        with st.expander('Data preview & offline notebook figures'):
            st.dataframe(filtered.head(100),width='stretch')
            pictures=sorted((ROOT/'outputs').glob('*.png'))
            if pictures:
                chosen=st.selectbox('Saved static figure',[p.name for p in pictures]);st.image(str(ROOT/'outputs'/chosen))
