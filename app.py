import streamlit as st
import numpy as np
import cv2
from math import log10, sqrt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import exposure
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.filters import threshold_otsu, threshold_local
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label
from io import BytesIO
import tempfile
import pandas as pd


class ImageQualityComparator:
    def __init__(self):
        self.reference_image = None
        self.degraded_image = None
        self.segmentation_masks = {}
        
    def load_images(self, reference_path, degraded_path):
        """Load reference and degraded images"""
        self.reference_image = cv2.cvtColor(cv2.imread(reference_path), cv2.COLOR_BGR2RGB)
        self.degraded_image = cv2.cvtColor(cv2.imread(degraded_path), cv2.COLOR_BGR2RGB)
        
        if self.reference_image is None or self.degraded_image is None:
            raise ValueError("Could not load one or both images")
            
        # Ensure both images have the same size
        if self.reference_image.shape != self.degraded_image.shape:
            self.degraded_image = cv2.resize(self.degraded_image, 
                                           (self.reference_image.shape[1], self.reference_image.shape[0]))
        
        return self.reference_image, self.degraded_image

    def process_image(self, image, method="basic"):
        """Apply various image processing methods"""
        processed = image.copy()
        
        if method == "basic":
            # Basic enhancement
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            processed = exposure.equalize_adapthist(processed, clip_limit=0.03) * 255
            processed = processed.astype(np.uint8)
            
        elif method == "contrast":
            # Contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            processed = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            
        elif method == "sharpen":
            # Sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
        elif method == "combined":
            # Combined enhancement
            # First denoise
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            # Then enhance contrast
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            processed = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            # Finally sharpen
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
        elif method == "contrast_sharp":
            # New optimized contrast and sharpening method
            
            # Step 1: Initial gentle denoising
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 3, 11)
            
            # Step 2: Enhanced contrast using CLAHE
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Using a smaller tile size and clip limit for more natural-looking results
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            cl = clahe.apply(l)
            
            # Step 3: Apply additional local contrast enhancement
            cl = exposure.equalize_adapthist(cl, clip_limit=0.01, kernel_size=127) * 255
            cl = cl.astype(np.uint8)
            
            # Merge channels back
            processed = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            
            # Step 4: Apply refined sharpening
            gaussian = cv2.GaussianBlur(processed, (0, 0), 2.0)
            processed = cv2.addWeighted(processed, 1.5, gaussian, -0.5, 0)
            
            # Step 5: Final contrast adjustment
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV).astype(float)
            hsv[:,:,1] = hsv[:,:,1] * 1.2  # Increase saturation
            hsv[:,:,2] = hsv[:,:,2] * 1.1  # Adjust brightness
            hsv = np.clip(hsv, 0, 255)
            processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        return processed.astype(np.uint8)

    def segment_objects(self, image, method="all"):
        """Perform object segmentation using various methods"""
        results = {}
        
        if method in ['threshold', 'all']:
            # Otsu's thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = threshold_otsu(gray)
            binary = (gray > thresh).astype(np.uint8) * 255
            
            # Adaptive thresholding
            adaptive_thresh = threshold_local(gray, block_size=35, offset=10)
            adaptive_binary = (gray > adaptive_thresh).astype(np.uint8) * 255
            
            results['threshold'] = {
                'otsu': binary,
                'adaptive': adaptive_binary
            }

        if method in ['kmeans', 'all']:
            # K-means segmentation
            pixel_values = image.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 3
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, 
                                          cv2.KMEANS_RANDOM_CENTERS)
            segmented_image = centers[labels.flatten()].reshape(image.shape)
            results['kmeans'] = segmented_image.astype(np.uint8)

        if method in ['watershed', 'all']:
            # Watershed segmentation
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers)
            results['watershed'] = markers

        if method in ['felzenszwalb', 'all']:
            # Felzenszwalb segmentation
            segments_fz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
            results['felzenszwalb'] = mark_boundaries(image, segments_fz)

        if method in ['slic', 'all']:
            # SLIC superpixels
            segments_slic = slic(image, n_segments=250, compactness=10, sigma=1,
                               start_label=1)
            results['slic'] = mark_boundaries(image, segments_slic)

        self.segmentation_masks = results
        return results

    def visualize_segmentation(self, method='all'):
        """Visualize segmentation results"""
        if not self.segmentation_masks:
            raise ValueError("No segmentation results available. Run segment_objects first.")

        if method == 'all':
            methods = list(self.segmentation_masks.keys())
        else:
            methods = [method]

        for method in methods:
            result = self.segmentation_masks[method]
            
            if method == 'threshold':
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(result['otsu'], cmap='gray')
                axes[0].set_title("Otsu's Thresholding")
                axes[0].axis('off')
                axes[1].imshow(result['adaptive'], cmap='gray')
                axes[1].set_title('Adaptive Thresholding')
                axes[1].axis('off')
            else:
                plt.figure(figsize=(6, 4))
                if method in ['felzenszwalb', 'slic']:
                    plt.imshow(result)
                elif method == 'watershed':
                    plt.imshow(result, cmap='nipy_spectral')
                else:
                    plt.imshow(result)
                plt.title(f'{method.upper()} Segmentation')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()

    def evaluate_quality(self, reference, test_image):
        """Calculate various quality metrics"""
        reference = np.asarray(reference, dtype=np.uint8)
        test_image = np.asarray(test_image, dtype=np.uint8)
        
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
            
        metrics = {
            'PSNR': self.calculate_psnr(reference, test_image),
            'SSIM': ssim(reference_gray, test_gray, data_range=255),
            'MSE': np.mean((reference.astype(float) - test_image.astype(float)) ** 2),
            'MAE': np.mean(np.abs(reference.astype(float) - test_image.astype(float)))
        }
        return metrics

    def calculate_psnr(self, reference, test_image):
        """Calculate PSNR"""
        mse = np.mean((reference.astype(float) - test_image.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def compare_improvements(self, processing_method="contrast_sharp"):
        """Compare quality before and after processing"""
        processed_image = self.process_image(self.degraded_image, method=processing_method)
        degraded_metrics = self.evaluate_quality(self.reference_image, self.degraded_image)
        processed_metrics = self.evaluate_quality(self.reference_image, processed_image)
        
        improvements = {}
        for metric in degraded_metrics.keys():
            if metric in ['PSNR', 'SSIM']:
                improvements[metric] = ((processed_metrics[metric] - degraded_metrics[metric]) / 
                                     degraded_metrics[metric] * 100)
            else:
                improvements[metric] = ((degraded_metrics[metric] - processed_metrics[metric]) / 
                                     degraded_metrics[metric] * 100)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(self.reference_image)
        axes[0].set_title('Reference Image')
        axes[0].axis('off')
        
        axes[1].imshow(self.degraded_image)
        axes[1].set_title('Degraded Image')
        axes[1].axis('off')
        
        axes[2].imshow(processed_image)
        axes[2].set_title(f'Processed Image\n({processing_method} method)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nQuality Metrics Comparison:")
        print("\nDegraded Image Metrics:")
        for metric, value in degraded_metrics.items():
            print(f"{metric}: {value:.3f}")
            
        print("\nProcessed Image Metrics:")
        for metric, value in processed_metrics.items():
            print(f"{metric}: {value:.3f}")
            
        print("\nImprovements:")
        for metric, improvement in improvements.items():
            print(f"{metric}: {improvement:+.2f}%")
            
        return processed_image, degraded_metrics, processed_metrics, improvements

    def process_and_segment(self, processing_method="contrast_sharp", segmentation_method="all"):
        """Process image and perform segmentation"""
        processed_image = self.process_image(self.degraded_image, method=processing_method)
        original_segments = self.segment_objects(self.degraded_image, method=segmentation_method)
        processed_segments = self.segment_objects(processed_image, method=segmentation_method)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(self.degraded_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(processed_image)
        plt.title(f'Processed Image\n({processing_method})')
        plt.axis('off')
        
        if 'slic' in processed_segments:
            plt.subplot(133)
            plt.imshow(processed_segments['slic'])
            plt.title('Segmentation Result\n(SLIC)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return processed_image, original_segments, processed_segments
    def process_image(self, image, method="basic"):
        """Apply various image processing methods"""
        processed = image.copy()
        
        if method == "basic":
            # Basic enhancement
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            processed = exposure.equalize_adapthist(processed, clip_limit=0.03) * 255
            processed = processed.astype(np.uint8)
            
        elif method == "contrast":
            # Contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            processed = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            
        elif method == "sharpen":
            # Sharpening
            kernel = np.array([[-1,-1,-1],
                            [-1, 9,-1],
                            [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
        elif method == "hist_global":
            # Global Histogram Equalization
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2YUV)
            processed[:,:,0] = cv2.equalizeHist(processed[:,:,0])
            processed = cv2.cvtColor(processed, cv2.COLOR_YUV2RGB)
            
        elif method == "hist_adaptive":
            # Adaptive Histogram Equalization
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(processed)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            processed = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            
        elif method == "hist_contrast_limited":
            # Contrast Limited Adaptive Histogram Equalization (CLAHE)
            processed = exposure.equalize_adapthist(processed, clip_limit=0.03) * 255
            processed = processed.astype(np.uint8)
            
        elif method == "gamma_correction":
            # Gamma Correction
            gamma = 1.5  # Adjust gamma value as needed
            processed = np.power(processed / 255.0, gamma) * 255.0
            processed = processed.astype(np.uint8)
            
        return processed.astype(np.uint8)

    def plot_histogram(self, image, title="Histogram"):
        """Plot color histogram for an image"""
        colors = ('r', 'g', 'b')
        
        plt.figure(figsize=(10, 4))
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.6)
            
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.2)
        plt.show()

    def analyze_image(self, image, title="Image Analysis"):
        """Perform comprehensive image analysis"""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        plt.suptitle(title, fontsize=16)
        
        # Original Image
        axes[0,0].imshow(image)
        axes[0,0].set_title('Original')
        axes[0,0].axis('off')
        
        # RGB Channels
        for i, c in enumerate(['Red', 'Green', 'Blue']):
            axes[0,1].plot(cv2.calcHist([image], [i], None, [256], [0, 256]), 
                        color=c.lower(), alpha=0.6, label=c)
        axes[0,1].set_title('RGB Histogram')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.2)
        
        # Grayscale
        axes[0,2].imshow(gray, cmap='gray')
        axes[0,2].set_title('Grayscale')
        axes[0,2].axis('off')
        
        # Grayscale Histogram
        axes[0,3].hist(gray.ravel(), 256, [0, 256], color='gray', alpha=0.6)
        axes[0,3].set_title('Grayscale Histogram')
        axes[0,3].grid(True, alpha=0.2)
        
        # HSV Channels
        for i, (j, title) in enumerate(zip([0,1,2], ['Hue', 'Saturation', 'Value'])):
            axes[1,i].imshow(hsv[:,:,j], cmap='gray')
            axes[1,i].set_title(f'HSV - {title}')
            axes[1,i].axis('off')
        
        # HSV Histogram
        axes[1,3].hist(hsv[:,:,1].ravel(), 256, [0, 256], color='blue', alpha=0.6)
        axes[1,3].set_title('Saturation Histogram')
        axes[1,3].grid(True, alpha=0.2)
        
        # LAB Channels
        for i, (j, title) in enumerate(zip([0,1,2], ['Lightness', 'A', 'B'])):
            axes[2,i].imshow(lab[:,:,j], cmap='gray')
            axes[2,i].set_title(f'LAB - {title}')
            axes[2,i].axis('off')
        
        # Lightness Histogram
        axes[2,3].hist(lab[:,:,0].ravel(), 256, [0, 256], color='green', alpha=0.6)
        axes[2,3].set_title('Lightness Histogram')
        axes[2,3].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.show()

    def process_and_analyze(self, method="all"):
        """Process image with different methods and show analysis"""
        if method == "all":
            methods = ["basic", "hist_global", "hist_adaptive", "hist_contrast_limited", 
                    "gamma_correction", "contrast", "sharpen"]
        else:
            methods = [method]
        
        # Create figure for processed images
        n_methods = len(methods)
        fig_rows = (n_methods + 2) // 3  # +2 for original image
        fig, axes = plt.subplots(fig_rows, 3, figsize=(15, 5*fig_rows))
        axes = axes.ravel()
        
        # Plot original image
        axes[0].imshow(self.degraded_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Process and plot each method
        for i, method in enumerate(methods, 1):
            processed = self.process_image(self.degraded_image, method=method)
            axes[i].imshow(processed)
            axes[i].set_title(f'{method.replace("_", " ").title()}')
            axes[i].axis('off')
            
            # Analyze original and processed images
            print(f"\n=== Analysis for {method.replace('_', ' ').title()} ===")
            self.analyze_image(processed, title=f"Analysis for {method.replace('_', ' ').title()}")
        
        plt.tight_layout()
        plt.show()

def main():
    st.set_page_config(layout="wide", page_title="Image Quality Enhancement & Analysis")
    
    st.title("Image Quality Enhancement & Analysis Tool")
    
    # File uploaders in a row
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload Reference Image")
        reference_file = st.file_uploader("Choose reference image", type=['jpg', 'jpeg', 'png'])
    
    with col2:
        st.subheader("Upload Test Image")
        test_file = st.file_uploader("Choose test image", type=['jpg', 'jpeg', 'png'])
    
    if reference_file is not None and test_file is not None:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as ref_tmp:
            ref_tmp.write(reference_file.getvalue())
            reference_path = ref_tmp.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as test_tmp:
            test_tmp.write(test_file.getvalue())
            test_path = test_tmp.name
        
        # Initialize comparator
        comparator = ImageQualityComparator()
        comparator.load_images(reference_path, test_path)
        
        # Display original images
        st.header("Input Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(comparator.reference_image, caption="Reference Image")
        with col2:
            st.image(comparator.degraded_image, caption="Test Image")
        
        # Processing methods selection
        st.header("Image Enhancement")
        processing_methods = ["basic", "contrast", "sharpen", "combined", "contrast_sharp", 
                            "hist_global", "hist_adaptive", "hist_contrast_limited", "gamma_correction"]
        
        selected_methods = st.multiselect(
            "Select processing methods to compare",
            processing_methods,
            default=["contrast_sharp"]
        )
        
        if selected_methods:
            # Process images and store results
            all_metrics = {}
            all_improvements = {}
            processed_images = {}
            
            for method in selected_methods:
                with st.expander(f"{method.replace('_', ' ').title()} Method Results"):
                    # Process image and get metrics
                    processed_image, degraded_metrics, processed_metrics, improvements = (
                        comparator.compare_improvements(processing_method=method)
                    )
                    
                    # Store results
                    all_metrics[method] = processed_metrics
                    all_improvements[method] = improvements
                    processed_images[method] = processed_image
                    
                    # Display processed image
                    st.image(processed_image, caption=f"Processed Image ({method})")
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Quality Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': list(processed_metrics.keys()),
                            'Value': list(processed_metrics.values())
                        })
                        st.table(metrics_df)
                    
                    with col2:
                        st.subheader("Improvements")
                        improvements_df = pd.DataFrame({
                            'Metric': list(improvements.keys()),
                            'Improvement (%)': [f"{v:+.2f}%" for v in improvements.values()]
                        })
                        st.table(improvements_df)
                    
                    # Histogram Analysis
                    st.subheader("Histogram Analysis")
                    comparator.analyze_image(processed_image)
                    st.pyplot(plt)
                    
                    # Segmentation Analysis
                    st.subheader("Segmentation Analysis")
                    processed_segments = comparator.segment_objects(processed_image)
                    comparator.segmentation_masks = processed_segments
                    comparator.visualize_segmentation()
                    st.pyplot(plt)
            
            # Comparative Analysis
            st.header("Comparative Analysis")
            
            # Metrics comparison table
            metrics = ['PSNR', 'SSIM', 'MSE', 'MAE']
            comparison_data = []
            for method in selected_methods:
                row = {'Method': method}
                row.update({metric: all_metrics[method][metric] for metric in metrics})
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.subheader("Quality Metrics Comparison")
            st.table(comparison_df)
            
            # Improvements comparison
            st.subheader("Improvements Comparison")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, metric in enumerate(metrics):
                improvements = [all_improvements[method][metric] for method in selected_methods]
                
                axes[idx].bar(selected_methods, improvements)
                axes[idx].set_title(f'{metric} Improvement')
                axes[idx].set_ylabel('Percentage Improvement')
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for i, v in enumerate(improvements):
                    axes[idx].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best performing methods
            st.subheader("Best Performing Methods")
            best_methods = []
            for metric in metrics:
                best_method = max(selected_methods, 
                                key=lambda x: all_improvements[x][metric])
                improvement = all_improvements[best_method][metric]
                best_methods.append({
                    'Metric': metric,
                    'Best Method': best_method,
                    'Improvement': f"{improvement:+.2f}%"
                })
            
            best_df = pd.DataFrame(best_methods)
            st.table(best_df)

if __name__ == "__main__":
    main()