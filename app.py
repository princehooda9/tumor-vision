import gradio as gr
import test_brats
import render_interactive_3d
import os

def run_omni_med_pipeline():
    # 1. Run the deep learning inference and generate 2D slices / NIfTI files
    print("Triggering Omni-Med AI Engine...")
    test_brats.run_evaluation()
    
    # 2. Run the 3D renderer to generate the Neon HTML file
    print("Triggering 3D Mesh Renderer...")
    render_interactive_3d.generate_neon_3d()
    
    # 3. Load the 2D PNG
    img_path = "prediction_multiclass_clear.png"
    
    # 4. Load the 3D HTML content directly into an iframe string
    html_file = "interactive_tumor_neon.html"
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    iframe_html = f"""
    <div style="height: 600px; width: 100%; border: 2px solid #00FFFF; border-radius: 10px; overflow: hidden;">
        <iframe srcdoc="{html_content.replace('"', '&quot;')}" style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    """
    
    return img_path, iframe_html

# ==========================================
# 🎨 BUILD THE WEB INTERFACE
# ==========================================
with gr.Blocks(theme=gr.themes.Monochrome()) as omni_med_app:
    gr.Markdown("<h1 style='text-align: center; color: #00FFFF;'>🧠 Omni-Med AI Diagnostic Platform</h1>")
    gr.Markdown("<p style='text-align: center;'>Automated 3D Brain Tumor Segmentation powered by Swin-UNETR.</p>")
    
    with gr.Row():
        analyze_btn = gr.Button("⚡ Analyze Random Patient File", variant="primary")
        
    with gr.Row():
        # Display the 2D Diagnostic PNG
        output_image = gr.Image(label="2D Cross-Section Analysis")
        
    with gr.Row():
        # Display the interactive 3D HTML
        output_3d = gr.HTML(label="Interactive 3D Structure")
        
    # Connect the button to the function
    analyze_btn.click(
        fn=run_omni_med_pipeline,
        inputs=[],
        outputs=[output_image, output_3d]
    )

if __name__ == "__main__":
    print("🚀 Launching Omni-Med Web Interface...")
    omni_med_app.launch(inbrowser=True)