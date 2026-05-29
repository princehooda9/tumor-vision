import nibabel as nib
import plotly.graph_objects as go
from skimage.measure import marching_cubes
import numpy as np

def generate_neon_3d():
    print("📂 Loading NIfTI volumes...")
    try:
        tumor_data = nib.load("omni_med_prediction.nii.gz").get_fdata()
        brain_data = nib.load("omni_med_flair.nii.gz").get_fdata()
    except FileNotFoundError:
        print("❌ Missing NIfTI files! Run test_brats.py first.")
        return

    fig = go.Figure()

    # ==========================================
    # 🧠 BRIGHTER BRAIN GHOST
    # ==========================================
    if np.sum(brain_data > 20) > 0:
        verts_b, faces_b, _, _ = marching_cubes(brain_data, level=20)
        fig.add_trace(go.Mesh3d(
            x=verts_b[:, 0], y=verts_b[:, 1], z=verts_b[:, 2],
            i=faces_b[:, 0], j=faces_b[:, 1], k=faces_b[:, 2],
            color='#ffffff', 
            opacity=0.15, # 🌟 Increased from 0.03 to 0.15 for much better visibility
            name='Brain Tissue (FLAIR)',
            hoverinfo='skip', 
            showlegend=True
        ))

    # ==========================================
    # ⚡ NEON TUMOR LAYERS
    # ==========================================
    regions = [
        (tumor_data >= 1, '#FF10F0', 'Whole Tumor (WT)', 0.15),   # Neon Pink
        (tumor_data >= 2, '#39FF14', 'Tumor Core (TC)', 0.35),    # Neon Green
        (tumor_data == 3, '#00FFFF', 'Enhancing Tumor (ET)', 1.0) # Neon Cyan
    ]

    for mask, color, name, opacity in regions:
        if np.sum(mask) > 0:
            verts, faces, _, _ = marching_cubes(mask, level=0.5)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=opacity, name=name,
                lighting=dict(ambient=0.6, diffuse=0.9, specular=0.8, roughness=0.1)
            ))

    # ==========================================
    # 📊 UI AND LEGEND POSITIONING
    # ==========================================
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
        title=dict(text="Omni-Med AI: Full Brain Neon Mapping", font=dict(size=22, color='#00FFFF')),
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        
        # 📌 Force the legend to the right side inside a framed box
        legend=dict(
            title=dict(text="Tissue Index", font=dict(color="white", size=16)),
            font=dict(color="white", size=14),
            bgcolor="rgba(20, 20, 20, 0.9)", # Dark grey box
            bordercolor="#00FFFF",           # Neon Cyan border
            borderwidth=1,
            x=0.85,       # Push to the right
            y=0.5,        # Center vertically
            xanchor="left",
            yanchor="middle"
        ),
        margin=dict(l=0, r=250, b=0, t=50) # Make room on the right for the legend
    )

    fig.write_html("interactive_tumor_neon.html")
    print("✅ Neon Render Complete! Open 'interactive_tumor_neon.html'.")

if __name__ == "__main__":
    generate_neon_3d()