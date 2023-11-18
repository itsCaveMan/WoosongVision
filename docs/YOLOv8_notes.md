
    results = A class for storing and manipulating inference results.
    Attribute	Type	                Description
    orig_img	numpy.ndarray	        The original image as a numpy array.
    orig_shape	tuple	                The original image shape in (height, width) format.
    boxes	    Boxes, optional	        A Boxes object containing the detection bounding boxes.
    masks	    Masks, optional	        A Masks object containing the detection masks.
    probs	    Probs, optional	        A Probs object containing probabilities of each class for classification task.
    keypoints	Keypoints, optional	    A Keypoints object containing detected keypoints for each object.
    speed	    dict	                A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
    names	    dict	                A dictionary of class names.
    path	    str	                    The path to the image file.



    results[0] = {
        cpu() -  Return a copy of the Results object with all tensors on CPU memory.
        __len__() - Return the number of detections in the Results object.
        cuda() - Return a copy of the Results object with all tensors on GPU memory.
        numpy() - Return a copy of the Results object with all tensors as numpy arrays.

        boxes - A class for storing and manipulating detection boxes.
        {
            Attributes:
            xyxy	Tensor | ndarray	The boxes in xyxy format.
            conf	Tensor | ndarray	The confidence values of the boxes.
            cls	    Tensor | ndarray	The class values of the boxes.
            id	    Tensor | ndarray	The track IDs of the boxes (if available).
            xywh	Tensor | ndarray	The boxes in xywh format.
            xyxyn	Tensor | ndarray	The boxes in xyxy format normalized by original image size.
            xywhn	Tensor | ndarray	The boxes in xywh format normalized by original image size.
            data	Tensor	            The raw bboxes tensor (alias for boxes).

            Methods:
            cpu	    Move the object to CPU memory.
            numpy	Convert the object to a numpy array.
            cuda	Move the object to CUDA memory.
            to	    Move the object to the specified device.

            Properties:
            xywh    Return the boxes in xywh format.
            xyxy    Return the boxes in xyxy format.
            xywhn   Return the boxes in xywh format normalized by original image size.
            xyxyn   Return the boxes in xyxy format normalized by original image size.
            id      Return the track IDs of the boxes (if available).
            conf    Return the confidence values of the boxes.
            cls     Return the class values of the boxes.
         }

        masks - A class for storing and manipulating detection masks.
        {
            Attributes:
            xy      A list of segments (in pixel coordinates.)
            xyn     A list of segments (in normalized coordinates.)

            Methods:
            cpu	    Returns the masks tensor on CPU memory.
            numpy	Returns the masks tensor as a numpy array.
            cuda	Returns the masks tensor on GPU memory.

            orig_shape - (1280, 1080)
            shape - ([1, 640, 544])
            names - {0: 'general_trashbag', 1: 'other_trashbag'}
        }
    }

    Pixel cordinates:
    Normalized cordinates:


    frame_history = [
        # [{...}] # previous frame result
        # [{...}] # previous frame result
        # [{...}] # previous frame result
    ]
    
    frame_results = [
        # {...} # 1 detected object
        {
            'id': 0,
            'box': {
                'xyxy': [0, 0, 0, 0],
                'xywh': [0, 0, 0, 0],
            },
            # 'mask': {
            #     'total_size_px': 12000,
            #     'points': [...],
            # }
        }
    ]




















