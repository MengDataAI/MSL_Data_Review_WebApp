
##################################################################################################
# the input is patient_df dataframe that contain data for a single patient, and 
# the output is line plot 
#
##################################################################################################
def plot_patient_line(patient_df):
    # Create Plotly figure
    fig = go.Figure()

    # Colorblind-friendly colors for patient lines (more vibrant)
    colors = [
        'rgb(230, 97, 1)',    # orange
        'rgb(94, 60, 153)',   # blue-purple
        'rgb(60, 180, 75)',   # green
        'rgb(214, 39, 40)',   # red
        'rgb(0, 114, 178)',   # blue
        'rgb(213, 94, 0)',    # dark orange
        'rgb(204, 121, 167)', # pink
        'rgb(230, 159, 0)',   # yellow-orange
        'rgb(86, 180, 233)',  # light blue
        'rgb(0, 158, 115)',   # teal
        'rgb(240, 228, 66)',  # yellow
        'rgb(0, 0, 0)'        # black
    ]

    # Add line traces for each patient
    for i, patient_id in enumerate(patient_df['PID'].unique()):
        patient_data = patient_df[patient_df['PID'] == patient_id]
        fig.add_trace(go.Scatter(
            x=patient_data['RESULT_POST_TRANSPLANT'],
            y=patient_data['RES_RESULT_NUMERIC'],
            mode='lines+markers',
            name=f'Patient {patient_id}',
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(symbol='circle', size=8, opacity=0.8, color=colors[i % len(colors)]),
            hovertemplate='<b>Patient %{fullData.name}</b><br>' +
                         'Days Post-Transplant: %{x}<br>' +
                         'AlloSure Score: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

    # Add alternating background shading for intervals
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    for i, (start, end) in enumerate(intervals):
        color = 'lightgray' if i % 2 == 0 else 'white'
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=0.35,
            layer="below",
            line_width=0
        )

    # Add horizontal reference line with annotation
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="darkred",
        line_width=3,
        annotation_text="1% threshold",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="darkred"
    )

    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>High Risk Patients with Subsequent AlloSure Score ≥ 1%</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=22)
        ),
        xaxis_title='<b>Days Post-Transplant</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            range=[0, 395],
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gainsboro',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            title='<b>Patient ID</b>',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=14, family='Arial'),
            bordercolor='gray',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=1000,
        height=540,
        margin=dict(l=100, r=40, t=80, b=60),
        hovermode='closest'
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')

    return fig

# test
patient_df = first_year_df[first_year_df['PID']=="P00000010YUU"]
plot_patient_line(patient_df)