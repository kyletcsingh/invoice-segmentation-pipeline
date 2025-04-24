PDF Receipt & Invoice Segmentation Pipeline  
-------------------------------------------  
Author: Kyle TC-Singh  
Version: 1.0  
Date: 04/2025  

OVERVIEW:  
This project provides a complete pipeline for processing scanned PDF files that contain one or more receipts, invoices, or ticket elements per page. The goal is to intelligently segment each page into its individual components and export them as separate PDF pages, ready for structured document intelligence analysis or data extraction.

KEY FEATURES:  
• Automatically detects and isolates multiple receipts/invoices per page  
• Handles both stacked and side-by-side layouts  
• Detects full-page documents and preserves them intact  
• Uses contour detection with dynamic kernel sizing  
• Merges related fragments (e.g., headers split from main body)  
• Produces high-resolution debug output with visual bounding boxes (optional)  
• Logs detection metadata in JSONL format (optional)  

PIPELINE LOGIC:  
1. **PDF to Image Conversion**: Each page is converted into a high-resolution image.  
2. **Contour Detection**: Edges and contours are analyzed to identify box-like regions using OpenCV.  
3. **Filtering & Grouping**: Small noise is filtered, and vertically aligned components are grouped.  
4. **Dominant Box Overrides**: Pages with large elements or multiple large boxes are identified as "full page" to prevent over-splitting.  
5. **Smart Merging**: Small fragments near larger components (e.g., titles or rotated tables) are merged back for accuracy.  
6. **Final Safety Pass**: Remaining small boxes are guaranteed to be attached to the nearest larger box.  
7. **Output Generation**:  
   • A clean "split" PDF with one element per page  
   • An optional "debug" PDF with visual bounding boxes (color-coded)  
   • An optional `.jsonl` log file with box coordinates, dimensions, and metadata  

CONFIGURATION:  
- Input Folder: `PDFs/Training Data`  
- Output Folder: `OpenCV/New Era Output 2`  
- Toggle debug mode by setting `save_debug_output = True` or `False`  

DEPENDENCIES:  
- Python 3.8+  
- OpenCV (`opencv-python`)  
- Pillow  
- pdf2image  
- Poppler (required for `pdf2image` backend)  

USAGE:  
Simply run the script after placing your PDFs in the input folder. Adjust configuration parameters if needed.  
Output PDFs will be saved in the output folder, either as individual receipts/invoices or as debug-annotated versions.

CONTACT:  
For questions or collaboration, please contact kyle.tcs@outlook.com
