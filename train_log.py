import os
import cv2
import numpy as np
import torch
from gaussian_renderer import render
import matplotlib.pyplot as plt
from matplotlib import cm



def visualize_skew_direction(
        gaussians, camera, pipe, background,
        output_path,          # …/skew_direction_iter_XXXXX.png
        iteration,
        threshold=None,       # None → no umbral, float → umbral en L2‐norm
        colormap='viridis'    # Mapa de color más atractivo
    ):
    """
    Renderiza una imagen donde el color de cada punto codifica la dirección
    del vector skew (x→R, y→G, z→B).  Si se pasa `threshold`, también se
    genera una segunda imagen en la que los Gaussians con ||skew||<threshold
    se vuelven totalmente transparentes.
    
    Parameters:
    -----------
    gaussians: objeto Gaussians que contiene los parámetros
    camera: cámara para la visualización
    pipe: pipeline de renderizado
    background: color de fondo
    output_path: ruta donde guardar la imagen
    iteration: número de iteración actual
    threshold: umbral para filtrar gaussianas con skew de magnitud baja
    colormap: nombre del mapa de colores de matplotlib para codificar direcciones
    """
    with torch.no_grad():
        feat_dc_orig   = gaussians._features_dc.data.clone()
        feat_rest_orig = gaussians._features_rest.data.clone()
        opacity_orig   = gaussians._opacity.data.clone()

        try:
            skews = gaussians._skews  
        except AttributeError:
            try:
                skews = gaussians.get_skews        
            except:
                print("ERROR: No se pudo acceder a los skews. Comprueba el nombre del atributo/método.")
                return None
        
        max_abs = torch.clamp(skews.abs().max(dim=1, keepdim=True)[0], 1e-5)
        rgb_dir = (skews / max_abs) * 0.5 + 0.5       # [-1,1]→[0,1]
        
        if colormap != 'rgb':
            dirs_np = rgb_dir.cpu().numpy()
            norm_dirs = np.linalg.norm(dirs_np, axis=1)
            max_norm = np.max(norm_dirs) if np.max(norm_dirs) > 0 else 1.0
            norm_dirs = norm_dirs / max_norm
            
            cmap = plt.get_cmap(colormap)
            colored = torch.tensor(cmap(norm_dirs)[:, :3], device=skews.device, dtype=skews.dtype)
            gaussians._features_dc.data = colored.view(-1, 1, 3)
        else:
            gaussians._features_dc.data = rgb_dir.view(-1, 1, 3)
            
        gaussians._features_rest.data = torch.zeros_like(feat_rest_orig)

        def _render_and_save(suffix, extra_title=""):
            img = torch.clamp(
                    render(camera, gaussians, pipe, background)["render"],
                    0., 1.)
            np_img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()

            h, w = np_img.shape[:2]
            title = np.zeros((50, w, 3), np.uint8)
            txt = f"Skew Direction {extra_title}- Iter {iteration}"
            
            # Fondo sombreado para el título
            cv2.rectangle(title, (0, 0), (w, 50), (30, 30, 30), -1)
            cv2.putText(title, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            final = np.vstack([title, np_img])

            path = (output_path if suffix == "" else
                    output_path.replace(".png", f"_{suffix}.png"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
            return img

        full_img = _render_and_save("")
        imgs_out = {"full": full_img}

        if threshold is not None:
            mag = torch.norm(skews, dim=1)                 
            mask_low = mag < threshold
            gaussians._opacity.data[mask_low] = 0.0        
            thresh_img = _render_and_save(
                f"thr_{threshold:.3f}",
                extra_title=f"(thr={threshold:.3f}) "
            )
            imgs_out["thresholded"] = thresh_img
            gaussians._opacity.data = opacity_orig.clone()

        gaussians._features_dc.data   = feat_dc_orig
        gaussians._features_rest.data = feat_rest_orig

        return imgs_out 



def visualize_skew_sensitivity(
        gaussians, camera, pipe, background,
        output_path,         
        iteration,
        colormap='plasma'):  
    """
    Renderiza un heat-map de la sensibilidad del skew con colores mejorados.
    
    Parameters:
    -----------
    gaussians: objeto Gaussians que contiene los parámetros
    camera: cámara para la visualización
    pipe: pipeline de renderizado
    background: color de fondo
    output_path: ruta donde guardar la imagen
    iteration: número de iteración actual
    colormap: nombre del mapa de colores de matplotlib para el heatmap
    """
    with torch.no_grad():
        feat_dc_orig   = gaussians._features_dc.data.clone()
        feat_rest_orig = gaussians._features_rest.data.clone()

        try:
            sens = gaussians._skew_sensitivity
        except AttributeError:
            try:
                sens = gaussians.get_skew_sensitivity
                if callable(sens):
                    sens = sens()
            except AttributeError:
                try:
                    sens = gaussians.get_skew_sensitivity()
                except (AttributeError, TypeError):
                    try:
                        sensitivity_attrs = [attr for attr in dir(gaussians) 
                                           if 'sensitivity' in attr.lower() and not attr.startswith('__')]
                        if sensitivity_attrs:
                            sens_attr = sensitivity_attrs[0]
                            sens = getattr(gaussians, sens_attr)
                            if callable(sens):
                                sens = sens()
                        else:
                            if hasattr(gaussians, '_skews'):
                                skews = gaussians._skews
                            elif hasattr(gaussians, 'get_skews'):
                                skews = gaussians.get_skews
                                if callable(skews):
                                    skews = skews()
                            else:
                                return None
                            
                            sens = torch.norm(skews, dim=1, keepdim=True)
                    except Exception as e:
                        return None
                
        if isinstance(sens, np.ndarray):
            sens = torch.tensor(sens, device=feat_dc_orig.device, dtype=feat_dc_orig.dtype)
        
        sens = sens.view(-1, 1) 
        
        if torch.isnan(sens).any() or torch.isinf(sens).any():
            sens = torch.nan_to_num(sens, nan=0.0, posinf=0.0, neginf=0.0)

        vmin, vmax = sens.min().item(), sens.max().item()
        if abs(vmax - vmin) < 1e-6:
            sens = sens + torch.rand_like(sens) * 1e-6
            vmin, vmax = sens.min().item(), sens.max().item()
            
        norm = (sens - vmin) / (vmax - vmin + 1e-8)    # 0…1
        
        cmap = plt.get_cmap(colormap)
        norm_np = norm.cpu().numpy().flatten()
        colors = cmap(norm_np)[:, :3]  
        rgb = torch.tensor(colors, device=sens.device, dtype=feat_dc_orig.dtype)
        
        gaussians._features_dc.data = rgb.view(-1, 1, 3)
        gaussians._features_rest.data = torch.zeros_like(feat_rest_orig)

        img = torch.clamp(
                render(camera, gaussians, pipe, background)["render"],
                0., 1.)
        np_img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()

        h, w = np_img.shape[:2]
        bar_h = 60
        colorbar = np.zeros((bar_h, w, 3), np.uint8)
        
        for i in range(w):
            p = i / (w-1)
            color = cmap(p)[:3]  
            colorbar[:bar_h-20, i] = [int(c*255) for c in color]

        cv2.rectangle(colorbar, (0, bar_h-20), (w, bar_h), (30, 30, 30), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(colorbar, f"Min {vmin:.3f}", (10, bar_h-5),
                    font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        txt = f"Max {vmax:.3f}"
        cv2.putText(colorbar, txt, (w-120, bar_h-5),
                    font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        title = np.zeros((50, w, 3), np.uint8)
        cv2.rectangle(title, (0, 0), (w, 50), (30, 30, 30), -1)
        txt_t = f"Skew Sensitivity – Iter {iteration} (μ={sens.mean().item():.3f})"
        cv2.putText(title, txt_t, (10, 30), font, 0.7, (255,255,255), 1, cv2.LINE_AA)

        final = np.vstack([title, np_img, colorbar])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        gaussians._features_dc.data   = feat_dc_orig
        gaussians._features_rest.data = feat_rest_orig
        return {"full": img}  


def visualize_skew_magnitude(
        gaussians, camera, pipe, background,
        output_path,          
        iteration,
        colormap='inferno',   
        log_scale=False):     
    """
    Renderiza un heat-map de la magnitud del vector skew.
    
    Parameters:
    -----------
    gaussians: objeto Gaussians que contiene los parámetros
    camera: cámara para la visualización
    pipe: pipeline de renderizado
    background: color de fondo
    output_path: ruta donde guardar la imagen
    iteration: número de iteración actual
    colormap: nombre del mapa de colores de matplotlib para el heatmap
    log_scale: si es True, aplica escala logarítmica para mejor visualización
    """
    with torch.no_grad():
        feat_dc_orig   = gaussians._features_dc.data.clone()
        feat_rest_orig = gaussians._features_rest.data.clone()

        try:
            skews = gaussians._skews                      # (N,3)
        except AttributeError:
            try:
                skews = gaussians.get_skews       
            except:
                print("ERROR: No se pudo acceder a los skews. Comprueba el nombre del atributo/método.")
                return None
        
        magnitude = torch.norm(skews, dim=1, keepdim=True)  # (N,1)
        
        if log_scale and magnitude.max() > 0:
            epsilon = 1e-5
            magnitude = torch.log(magnitude + epsilon)
        
        vmin, vmax = magnitude.min().item(), magnitude.max().item()
        norm = (magnitude - vmin) / (vmax - vmin + 1e-8)  # 0…1
        
        cmap = plt.get_cmap(colormap)
        norm_np = norm.cpu().numpy().flatten()
        colors = cmap(norm_np)[:, :3]
        rgb = torch.tensor(colors, device=skews.device, dtype=feat_dc_orig.dtype)
        
        gaussians._features_dc.data = rgb.view(-1, 1, 3)
        gaussians._features_rest.data = torch.zeros_like(feat_rest_orig)

        img = torch.clamp(
                render(camera, gaussians, pipe, background)["render"],
                0., 1.)
        np_img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()

        mean_mag = magnitude.mean().item()
        median_mag = torch.median(magnitude).item()
        
        h, w = np_img.shape[:2]
        bar_h = 60
        colorbar = np.zeros((bar_h, w, 3), np.uint8)
        
        for i in range(w):
            p = i / (w-1)
            color = cmap(p)[:3] 
            colorbar[:bar_h-20, i] = [int(c*255) for c in color]
        
        cv2.rectangle(colorbar, (0, bar_h-20), (w, bar_h), (30, 30, 30), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_txt = "Log scale" if log_scale else "Linear scale"
        min_txt = f"Min: {vmin:.3f}"
        max_txt = f"Max: {vmax:.3f}"
        
        cv2.putText(colorbar, min_txt, (10, bar_h-5),
                    font, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(colorbar, max_txt, (w-120, bar_h-5),
                    font, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(colorbar, scale_txt, (w//2-40, bar_h-5),
                    font, 0.45, (255,255,255), 1, cv2.LINE_AA)

        title_h = 70
        title = np.zeros((title_h, w, 3), np.uint8)
        cv2.rectangle(title, (0, 0), (w, title_h), (30, 30, 30), -1)
        
        title_txt = f"Skew Magnitude – Iter {iteration}"
        stats_txt = f"Mean: {mean_mag:.3f} | Median: {median_mag:.3f}"
        
        cv2.putText(title, title_txt, (10, 25), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(title, stats_txt, (10, 55), font, 0.6, (220,220,220), 1, cv2.LINE_AA)

        final = np.vstack([title, np_img, colorbar])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        # restaurar
        gaussians._features_dc.data   = feat_dc_orig
        gaussians._features_rest.data = feat_rest_orig
        return {"full": img}



def _log_to_tb(tag, tensor, step, tb_writer):
    if tensor is None:
        return
    tb_img = torch.clamp(tensor.detach().cpu().float(), 0, 1)  # (3,H,W)
    tb_writer.add_image(tag, tb_img, global_step=step)

def visualize_all_skew_properties(
        gaussians, camera, pipe, background,
        output_folder,      
        iteration,
        tb_writer=None):      
    """
    Ejecuta las tres visualizaciones (dirección, sensibilidad y magnitud) 
    y opcionalmente las envía a TensorBoard.
    """
    os.makedirs(output_folder, exist_ok=True)
    results = {}
    
    skew_dir_path = os.path.join(output_folder, f"skew_direction_iter_{iteration:06d}.png")
    try:
        skew_dir_imgs = visualize_skew_direction(
            gaussians, camera, pipe, background,
            skew_dir_path, iteration,
            threshold=None,  
            colormap='rainbow'  
        )
        if skew_dir_imgs:
            results["direction"] = skew_dir_imgs["full"]
    except Exception as e:
        print(f"Error en visualización de dirección: {e}")
    
    skew_sens_path = os.path.join(output_folder, f"skew_sensitivity_iter_{iteration:06d}.png")
    try:
        skew_sens_image = visualize_skew_sensitivity(
            gaussians, camera, pipe, background, 
            skew_sens_path, iteration,
            colormap='plasma' 
        )
        if skew_sens_image:
            results["sensitivity"] = skew_sens_image["full"]
    except Exception as e:
        print(f"Error en visualización de sensibilidad: {e}")
    
    skew_mag_path = os.path.join(output_folder, f"skew_magnitude_iter_{iteration:06d}.png")
    try:
        skew_mag_image = visualize_skew_magnitude(
            gaussians, camera, pipe, background,
            skew_mag_path, iteration,
            colormap='inferno',  
            log_scale=False     
        )
        if skew_mag_image:
            results["magnitude"]  = skew_mag_image["full"]
    except Exception as e:
        print(f"Error en visualización de magnitud: {e}")
    
    if tb_writer and results:
        try:
            
            for key, img in results.items():
                _log_to_tb(f"skew/{key}", img, iteration, tb_writer)
            
            try:
                try:
                    skews = gaussians._skews
                except AttributeError:
                    try:
                        skews = gaussians.get_skews
                        if callable(skews):
                            skews = skews()
                    except AttributeError:
                        skews = None
                
                if skews is not None:
                    try:
                        tb_writer.add_histogram("skew/directions_x", skews[:, 0], global_step=iteration)
                        tb_writer.add_histogram("skew/directions_y", skews[:, 1], global_step=iteration)
                        tb_writer.add_histogram("skew/directions_z", skews[:, 2], global_step=iteration)
                        
                        magnitude = torch.norm(skews, dim=1)
                        tb_writer.add_histogram("skew/magnitude", magnitude, global_step=iteration)
                        tb_writer.add_scalar("skew/mean_magnitude", magnitude.mean(), global_step=iteration)
                        tb_writer.add_scalar("skew/max_magnitude", magnitude.max(), global_step=iteration)
                    except Exception as e:
                        print(f"Error adding skew magnitude metrics: {e}")
                
                try:
                    try:
                        sens = gaussians._skew_sensitivity
                    except AttributeError:
                        try:
                            sens = gaussians.get_skew_sensitivity
                            if callable(sens):
                                sens = sens()
                        except AttributeError:
                            sens = None
                    
                    if sens is not None:
                        if hasattr(sens, "squeeze"):
                            sens = sens.squeeze()
                        tb_writer.add_histogram("skew/sensitivity", sens, global_step=iteration)
                        tb_writer.add_scalar("skew/mean_sensitivity", sens.mean(), global_step=iteration)
                        tb_writer.add_scalar("skew/max_sensitivity", sens.max(), global_step=iteration)
                        tb_writer.add_scalar("skew/min_sensitivity", sens.min(), global_step=iteration)
                except Exception as e:
                    print(f"Error adding skew sensitivity metrics: {e}")
                    
            except Exception as e:
                print(f"Error al loguear estadísticas: {e}")
        except Exception as e:
            print(f"Error en tensorboard: {e}")
    tb_writer.flush()
    return results


def diagnose_gaussians(gaussians, iteration, tb_writer=None, grad_check=True):
    """
    Diagnose Gaussian model parameters to detect potential issues like NaN, Inf values.
    Only logs tables to TensorBoard without printing anything.
    
    Args:
        gaussians: The Gaussian model
        iteration: Current training iteration (for logging)
        tb_writer: TensorBoard writer for logging diagnostics as tables
        grad_check: Whether to check gradients as well
    """
    if tb_writer is None:
        return False
    
    param_names = ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation', '_skews', '_skew_sensitivity']
    
    has_problem = False
    
    tb_data = {
        "Parameter": [],
        "Shape": [],
        "Min": [],
        "Max": [],
        "Mean": [],
        "Has_NaN": [],
        "Has_Inf": []
    }
    
    if grad_check:
        tb_grad_data = {
            "Parameter": [],
            "Min": [],
            "Max": [],
            "Mean": [],
            "Norm": [],
            "Has_NaN": [],
            "Has_Inf": []
        }

    for name in param_names:
        if not hasattr(gaussians, name):
            continue
            
        param = getattr(gaussians, name)
        if param is None:
            continue
            
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        
        if param.numel() > 0:
            min_val = param.min().item()
            max_val = param.max().item()
            mean_val = param.mean().item()
        else:
            min_val = max_val = mean_val = 0
        
        tb_data["Parameter"].append(name)
        tb_data["Shape"].append(str(param.shape))
        tb_data["Min"].append(f"{min_val:.4f}")
        tb_data["Max"].append(f"{max_val:.4f}")
        tb_data["Mean"].append(f"{mean_val:.4f}")
        tb_data["Has_NaN"].append(str(has_nan))
        tb_data["Has_Inf"].append(str(has_inf))
        
        if has_nan or has_inf:
            has_problem = True
        
        if grad_check and param.grad is not None:
            grad = param.grad
            grad_has_nan = torch.isnan(grad).any().item()
            grad_has_inf = torch.isinf(grad).any().item()
            
            if grad.numel() > 0:
                grad_min_val = grad.min().item()
                grad_max_val = grad.max().item()
                grad_mean_val = grad.mean().item()
                grad_norm = grad.norm().item()
            else:
                grad_min_val = grad_max_val = grad_mean_val = grad_norm = 0
            
            tb_grad_data["Parameter"].append(f"{name}.grad")
            tb_grad_data["Min"].append(f"{grad_min_val:.4f}")
            tb_grad_data["Max"].append(f"{grad_max_val:.4f}")
            tb_grad_data["Mean"].append(f"{grad_mean_val:.4f}")
            tb_grad_data["Norm"].append(f"{grad_norm:.4f}")
            tb_grad_data["Has_NaN"].append(str(grad_has_nan))
            tb_grad_data["Has_Inf"].append(str(grad_has_inf))
            
            if grad_has_nan or grad_has_inf:
                has_problem = True
    
    state_vars = ['max_radii2D', 'xyz_gradient_accum', 'denom']
    for name in state_vars:
        if hasattr(gaussians, name):
            var = getattr(gaussians, name)
            if var is not None and isinstance(var, torch.Tensor):
                has_nan = torch.isnan(var).any().item()
                has_inf = torch.isinf(var).any().item()
                
                if var.numel() > 0:
                    min_val = var.min().item()
                    max_val = var.max().item()
                    mean_val = var.mean().item()
                else:
                    min_val = max_val = mean_val = 0
                
                tb_data["Parameter"].append(name)
                tb_data["Shape"].append(str(var.shape))
                tb_data["Min"].append(f"{min_val:.4f}")
                tb_data["Max"].append(f"{max_val:.4f}")
                tb_data["Mean"].append(f"{mean_val:.4f}")
                tb_data["Has_NaN"].append(str(has_nan))
                tb_data["Has_Inf"].append(str(has_inf))
                
                if has_nan or has_inf:
                    has_problem = True
    
    if hasattr(gaussians, 'get_scaling'):
        scaling = gaussians.get_scaling
        if scaling.numel() > 0:
            tb_data["Parameter"].append("Scaling (activated)")
            tb_data["Shape"].append(str(scaling.shape))
            tb_data["Min"].append(f"{scaling.min().item():.4f}")
            tb_data["Max"].append(f"{scaling.max().item():.4f}")
            tb_data["Mean"].append(f"{scaling.mean().item():.4f}")
            tb_data["Has_NaN"].append(str(torch.isnan(scaling).any().item()))
            tb_data["Has_Inf"].append(str(torch.isinf(scaling).any().item()))
    
    if hasattr(gaussians, 'get_opacity'):
        opacity = gaussians.get_opacity
        if opacity.numel() > 0:
            tb_data["Parameter"].append("Opacity (activated)")
            tb_data["Shape"].append(str(opacity.shape))
            tb_data["Min"].append(f"{opacity.min().item():.4f}")
            tb_data["Max"].append(f"{opacity.max().item():.4f}")
            tb_data["Mean"].append(f"{opacity.mean().item():.4f}")
            tb_data["Has_NaN"].append(str(torch.isnan(opacity).any().item()))
            tb_data["Has_Inf"].append(str(torch.isinf(opacity).any().item()))
    
    import pandas as pd
    
    params_df = pd.DataFrame(tb_data)
    tb_writer.add_text("diagnostics/parameters", params_df.to_markdown(), global_step=iteration)
    
    if grad_check:
        grads_df = pd.DataFrame(tb_grad_data)
        tb_writer.add_text("diagnostics/gradients", grads_df.to_markdown(), global_step=iteration)
    
    return has_problem



def calculate_gaussian_normals(gaussians, filter_anisotropy=True, anisotropy_threshold=4.0):
    """
    Calcula las normales de las Gaussianas usando el método del mínimo eigen-vector.
    
    Parameters:
    -----------
    gaussians: objeto Gaussians que contiene los parámetros
    filter_anisotropy: si True, filtra Gaussianas casi esféricas
    anisotropy_threshold: umbral de anisotropía (λ_max/λ_min)
    
    Returns:
    --------
    normals: tensor (N, 3) con las normales calculadas
    valid_mask: tensor (N,) booleano indicando cuáles son válidas
    """
    with torch.no_grad():
        scales = gaussians.get_scaling  # (N, 3)
        rotations = gaussians.get_rotation  # (N, 4) quaternions
        
        N = scales.shape[0]
        normals = torch.zeros((N, 3), device=scales.device, dtype=scales.dtype)
        valid_mask = torch.ones(N, device=scales.device, dtype=torch.bool)

        w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        
        R = torch.zeros((N, 3, 3), device=scales.device, dtype=scales.dtype)
        
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - w*x)
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        
        S_diag = torch.diag_embed(scales * scales)  # (N, 3, 3)
        Sigma = torch.bmm(torch.bmm(R, S_diag), R.transpose(-2, -1))  # (N, 3, 3)
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(Sigma)  # (N, 3), (N, 3, 3)
            
            min_indices = torch.argmin(eigenvals, dim=1)  # (N,)
            
            for i in range(N):
                normals[i] = eigenvecs[i, :, min_indices[i]]
            
            if filter_anisotropy:
                max_eigenvals = torch.max(eigenvals, dim=1)[0]
                min_eigenvals = torch.min(eigenvals, dim=1)[0]
                anisotropy = max_eigenvals / (min_eigenvals + 1e-8)
                valid_mask = anisotropy > anisotropy_threshold
                
        except Exception as e:
            print(f"Error calculando normales: {e}")
            normals = R[:, :, 2] 
        
        return normals, valid_mask


def visualize_gaussian_normals(
        gaussians, camera, pipe, background,
        output_path,          
        iteration,
        colormap='rainbow',  
        filter_anisotropy=True,
        anisotropy_threshold=4.0):
    """
    Renderiza una imagen donde el color de cada punto codifica la dirección
    de la normal de la Gaussiana (x→R, y→G, z→B).
    
    Parameters:
    -----------
    gaussians: objeto Gaussians que contiene los parámetros
    camera: cámara para la visualización
    pipe: pipeline de renderizado
    background: color de fondo
    output_path: ruta donde guardar la imagen
    iteration: número de iteración actual
    colormap: nombre del mapa de colores ('rgb' para RGB directo, otro para matplotlib)
    filter_anisotropy: si True, filtra Gaussianas casi esféricas
    anisotropy_threshold: umbral de anisotropía para filtrado
    """
    with torch.no_grad():
        feat_dc_orig   = gaussians._features_dc.data.clone()
        feat_rest_orig = gaussians._features_rest.data.clone()
        opacity_orig   = gaussians._opacity.data.clone()

        normals, valid_mask = calculate_gaussian_normals(
            gaussians, filter_anisotropy, anisotropy_threshold)
        
        camera_pos = camera.camera_center  
        gaussians_pos = gaussians.get_xyz 
        
        to_camera = camera_pos.unsqueeze(0) - gaussians_pos  # (N, 3)
        to_camera = to_camera / (torch.norm(to_camera, dim=1, keepdim=True) + 1e-8)
        
        dot_product = torch.sum(normals * to_camera, dim=1)  # (N,)
        flip_mask = dot_product < 0
        normals[flip_mask] = -normals[flip_mask]
        
        if colormap == 'rgb':
            rgb_dir = normals * 0.5 + 0.5
        else:
            x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
            
            azimuth = torch.atan2(y, x)  # [-π, π]
            elevation = torch.asin(torch.clamp(z, -1, 1))  # [-π/2, π/2]
            
            azimuth_norm = (azimuth + np.pi) / (2 * np.pi)  # [0, 1]
            elevation_norm = (elevation + np.pi/2) / np.pi   # [0, 1]
            
            color_index = (azimuth_norm + elevation_norm) / 2
            
            cmap = plt.get_cmap(colormap)
            colors_np = cmap(color_index.cpu().numpy())[:, :3]  # RGB sin alpha
            rgb_dir = torch.tensor(colors_np, device=normals.device, dtype=normals.dtype)
        
        if filter_anisotropy:
            gray_color = torch.tensor([0.5, 0.5, 0.5], device=rgb_dir.device, dtype=rgb_dir.dtype)
            rgb_dir[~valid_mask] = gray_color
        
        gaussians._features_dc.data = rgb_dir.view(-1, 1, 3)
        gaussians._features_rest.data = torch.zeros_like(feat_rest_orig)
        
        if filter_anisotropy:
            gaussians._opacity.data[~valid_mask] = gaussians._opacity.data[~valid_mask] * 0.3

        img = torch.clamp(
                render(camera, gaussians, pipe, background)["render"],
                0., 1.)
        np_img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()

        h, w = np_img.shape[:2]
        title_h = 70
        title = np.zeros((title_h, w, 3), np.uint8)
        
        valid_count = valid_mask.sum().item()
        total_count = len(valid_mask)
        
        cv2.rectangle(title, (0, 0), (w, title_h), (30, 30, 30), -1)
        
        main_txt = f"Gaussian Normals - Iter {iteration}"
        stats_txt = f"Valid: {valid_count}/{total_count} | Anisotropy thr: {anisotropy_threshold:.1f}"
        
        cv2.putText(title, main_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(title, stats_txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        legend_h = 60
        legend = np.zeros((legend_h, w, 3), np.uint8)
        cv2.rectangle(legend, (0, 0), (w, legend_h), (30, 30, 30), -1)
        
        if colormap == 'rgb':
            legend_txt = "Colors: X→Red, Y→Green, Z→Blue | Gray→Filtered"
        else:
            legend_txt = f"Colors: {colormap} (spherical mapping) | Gray→Filtered"
            
        cv2.putText(legend, legend_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend, "Normals oriented toward camera", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        final = np.vstack([title, np_img, legend])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        gaussians._features_dc.data   = feat_dc_orig
        gaussians._features_rest.data = feat_rest_orig
        gaussians._opacity.data = opacity_orig

        return {"full": img}



def visualize_all_gaussian_properties(
        gaussians, camera, pipe, background,
        output_folder,       
        iteration,
        tb_writer=None):      
    """
    Ejecuta todas las visualizaciones (skew + normales) y opcionalmente 
    las envía a TensorBoard.
    """
    os.makedirs(output_folder, exist_ok=True)
    results = {}
    
    normals_path = os.path.join(output_folder, f"gaussian_normals_iter_{iteration:06d}.png")

    try:
        normals_image = visualize_gaussian_normals(
            gaussians, camera, pipe, background,
            normals_path, iteration,
            colormap='rainbow', 
            filter_anisotropy=True,
            anisotropy_threshold=4.0
        )
        if normals_image:
            results["normals"] = normals_image["full"]
    except Exception as e:
        print(f"Error en visualización de normales: {e}")
    
    if tb_writer and "normals" in results:
        try:
            _log_to_tb("gaussian/normals", results["normals"], iteration, tb_writer)
            
            try:
                normals, valid_mask = calculate_gaussian_normals(gaussians)
                
                tb_writer.add_histogram("gaussian/normals_x", normals[:, 0], global_step=iteration)
                tb_writer.add_histogram("gaussian/normals_y", normals[:, 1], global_step=iteration)
                tb_writer.add_histogram("gaussian/normals_z", normals[:, 2], global_step=iteration)
                
                valid_ratio = valid_mask.float().mean().item()
                tb_writer.add_scalar("gaussian/normals_valid_ratio", valid_ratio, global_step=iteration)
                tb_writer.add_scalar("gaussian/normals_count", valid_mask.sum().item(), global_step=iteration)
                
            except Exception as e:
                print(f"Error al loguear estadísticas de normales: {e}")
        except Exception as e:
            print(f"Error en tensorboard para normales: {e}")
    
    if tb_writer:
        tb_writer.flush()
    
    return results