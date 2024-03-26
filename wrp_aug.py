import numpy as np
import cv2
import math
from scipy import ndimage

def _localTranslationWarp(image, warpedImg, centerX, centerY, radius, strength, sq_m_minus_c, m_minus_cX, m_minus_cY, interpolation) -> np.ndarray:
    sqRadius = radius*radius
    height, width, _ = warpedImg.shape
    #Bounding Box of the circle
    startX = int(centerX-radius)
    startY = int(centerY-radius)
    endX = int(centerX+radius)
    endY = int(centerY+radius)
    if(endX >= width): endX = width - 1
    if(endX >= height): endX = height - 1

    map = np.zeros((512, 512, 2), np.int8)

    #Iterate over the Bounding Box
    for col in range(startX, endX):
        for row in range(startY, endY):
            distanceX = (col - centerX)
            distanceY = (row - centerY)
            #Point inside Bounding Box?
            if(distanceX > 0.0-radius and distanceX < radius and
                distanceY > 0.0-radius and distanceY < radius):
                sqDistance = distanceX*distanceX + distanceY*distanceY
                #Point inside circle?
                if(sqDistance < sqRadius):
                    #sqDistance corresponds to the expression |x-c|^2 in the master's thesis
                    numerator = sqRadius - sqDistance
                    denominator = numerator + sq_m_minus_c / strength
                    
                    #denominator cannot be == 0 as numerator is always >= 0 and sq_m_minus_c is always > 0
                    fraction = numerator / denominator
                    sqFraction = fraction*fraction

                    sourceX = col - sqFraction * m_minus_cX 
                    sourceY = row - sqFraction * m_minus_cY 

                    if(sourceX < 0): sourceX = 0
                    if(sourceY < 0): sourceY = 0
                    #Minus 2 because the interpolation will need room for one more pixel in each direction
                    if(sourceX >= width): sourceX = width - 2
                    if(sourceY >= height): sourceY = height - 2

                    if(not interpolation):
                        warpedImg[row][col] = image[int(round(sourceY))][int(round(sourceX))]
                    else:
                        #bilinear interpolation
                        # A │       │     C │
                        #───0─────▲─┼───────0──
                        #   │     │ │       │
                        #   ◄─────x─┼───────►
                        #───┼─────┼─┼───────┼──
                        #   │     │ │       │
                        # B │     │ │     D │
                        #───0─────▼─┼───────0──   
                        x1 = min(int(sourceX), width-1)
                        y1 = min(int(sourceY), height-1)
                        x2 = min(int(sourceX+1), width-1)
                        y2 = min(int(sourceY+1), height-1)
                        A = image[y1, x1]
                        B = image[y2, x1]
                        C = image[y1, x2]
                        D = image[y2, x2]
                        weightA = (x2 - sourceX) * (y2 - sourceY)
                        weightB = (x2 - sourceX) * (sourceY - y1)
                        weightC = (sourceX - x1) * (y2 - sourceY)
                        weightD = (sourceX - x1) * (sourceY - y1)
                        
                        warpedImg[row][col] = weightA * A + weightB * B + weightC * C + weightD * D
                    
                    map[row][col] = (row - int(sourceY), col - int(sourceX))
                    
    return warpedImg, map

def _localTranslationWarpRemap(image, warpedImg, centerX, centerY, radius, strength, sq_m_minus_c, m_minus_cX, m_minus_cY, interpolation) -> np.ndarray:
    maskImg = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(maskImg, (centerX, centerY), int(radius+0.5), (255, 255, 255), -1)
    
    sqRadius = radius*radius
    height, width, _ = warpedImg.shape

    mapX = np.vstack([np.arange(width).astype(np.float32).reshape(1, -1)] * height)
    mapY = np.hstack([np.arange(height).astype(np.float32).reshape(-1, 1)] * width)

    distanceX = (mapX - centerX) 
    distanceY = (mapY - centerY)
    sqDistance = distanceX*distanceX + distanceY*distanceY
    numerator = sqRadius - sqDistance
    denominator = numerator + sq_m_minus_c / strength
    #Hack, to prevent division by 0. In this case it can happen as the whole size of the image is used
    denominator[np.abs(denominator) == 0.0] = 0.0000000000001
    fraction = numerator / denominator
    sqFraction = fraction*fraction

    sourceX = mapX - sqFraction * m_minus_cX 
    sourceY = mapY - sqFraction * m_minus_cY

    np.copyto(mapX, sourceX, where=(maskImg == 255))
    np.copyto(mapY, sourceY, where=(maskImg == 255))
    sourceX = sourceX.astype(np.float32)
    sourceY = sourceY.astype(np.float32)
    
    inter = cv2.INTER_LINEAR
    if not interpolation:
        inter = None

    warpedImg = cv2.remap(image, mapX, mapY, interpolation=inter)

    resultX = np.zeros(image.shape[:2], np.float32)
    resultX[maskImg==255] = sourceX[maskImg==255]
    resultY = np.zeros(image.shape[:2], np.float32)
    resultY[maskImg==255] = sourceY[maskImg==255]

    return warpedImg, resultX, resultY

def localTranslationWarp(image, radius, centerX, centerY, targetX, targetY, strength=1.0, interpolation=False, useRemap=True) -> np.ndarray:
    """Creates a warped version of the image. The warping occurs at the provided center in the
    direction of the target with a specified strength inside a circle with the provided radius. 
    This algorithm is based on the Master's Thesis "Interactive Image Warping" by Andreas Gustafsson.

    Parameters
    ----------
    image : `ndarray`
        Source image which will be warped.

    radius : `float`
        Radius of the circle inside which the warping happens. Has to be > 0.
    
    centerX : `float`
        x-Coordinate of the warping circles center.

    centerY: `float`
        y-Coordinate of the warping circles center.

    targetX : `float`
        x-Coordinate of the warping target.

    targetY: `float`
        y-Coordinate of the warping target.

    strength : `float`
        Strength of the warping effect. 
        By default the warping will have a strength of `radius`.
        Has to be > 0

    interpolation: `boolean`
        Activates bilinear interpolation.

    Returns
    -------
    localTranslationWarp : `ndarray`
        Returns the warped image. Warping occurs in the area of the provided circle 
        with the specified strength and direction.
    """

    # Initialising

    if(radius <= 0):
        raise TypeError("Radius has to be > 0.")
    if(strength <= 0):
        raise TypeError("Strength has to be > 0.")

    warpedImg = np.copy(image)

    #mX and mY corresponds to the vector m in the master's thesis. 
    mX = targetX
    mY = targetY

    #|m - center|
    m_minus_cX = (mX - centerX)
    m_minus_cY = (mY - centerY)

    #|m - center|^2
    sq_m_minus_c = m_minus_cX*m_minus_cX + m_minus_cY*m_minus_cY
    if(sq_m_minus_c == 0):
        sq_m_minus_c = 0.01
    
    _, _, channels = warpedImg.shape
    
    if(channels != 3):
        raise TypeError("Channels have to be 3 (for now).")

    if useRemap:
        return _localTranslationWarpRemap(image, warpedImg, centerX, centerY, radius, strength, sq_m_minus_c, m_minus_cX, m_minus_cY, interpolation)
    else:
        return _localTranslationWarp(image, warpedImg, centerX, centerY, radius, strength, sq_m_minus_c, m_minus_cX, m_minus_cY, interpolation)

def _getLargestSegmentationShapeInfo(segmentation):
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largestContour = None
    largestArea = -1
    largestMoments = None
    strength = -1
    centerX = -1
    centerY = -1
    enclosingRadius = -1
    r = -1

    for contour in contours:
        moments = cv2.moments(contour)
        if(largestArea < moments['m00']):
            largestArea = moments['m00']
            largestContour = contour
            largestMoments = moments

    if largestContour is not None:
        (x,y), enclosingRadius = cv2.minEnclosingCircle(largestContour)

        r = math.sqrt((largestArea / (math.pi))/3)

        strengthMax = min(2 * (r/(enclosingRadius*2)), 1)
        strengthMin = max(0.01, 1 - strengthMax)
        strength = np.random.uniform(strengthMin, strengthMax)

        centerX = int(largestMoments['m10']/largestMoments['m00'])
        centerY = int(largestMoments['m01']/largestMoments['m00'])
    
    return largestContour, largestArea, strength, centerX, centerY, enclosingRadius, r

class CircularWarping:
    """Applies a warping at every 360.0 / N degrees in the direction indicated by the flag 'outwards'. 
        For the positioning of the warping direction the shape in the segmentation mask is used. 

    Parameters
    ----------
    image : `ndarray`
        Source image which will be warped.
    
    segmentation : `ndarray`
        Segmentation mask of the source image.
    
    N : `int`
        Indicates the number of warpings in a circular arrangement.
    
    outwards: `boolean`
        Indicates if the warpings are going away from the warping center or toward the warping center.

    Returns
    -------
    WarpedImage : `np.ndarray`
        Returns a copy of the original image with the warpings applied.
    """

    def __init__(self, N = 8, outwards = True):
        self.N = N
        self.outwards = outwards
        self.lastMap = None

    def get_distance_and_direction(self, center, point):
        distance = math.sqrt((center['x'] - point['x'])**2 + (center['y'] - point['y'])**2)
        if distance <= 0:
            distance = 1
        direction = {'x': (point['x'] - center['x']) / distance, 'y': (point['y'] - center['y']) / distance}

        return {'distance': distance, 'direction': direction}

    def __call__(self, sample):
        image, segmentation = sample
        
        largestContour, largestArea, strength, centerX, centerY, radius, r = _getLargestSegmentationShapeInfo(cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY))
        
        if(self.outwards is False and largestArea < 50):
            #No 'healing' augmentation possible
            return image, segmentation
        if(largestContour is None):
            #No augmentation possible without a contour
            return image, segmentation

        contourLength = len(largestContour)
        stepSize = contourLength / self.N
        intersections = []

        # to get more flexibility, sample a random shift parameter to get not always the same starting point
        shift = np.random.uniform(0, stepSize)

        # sample N points from the contour
        for i in range(self.N):
            point = largestContour[(int(i * stepSize)+int(shift)) % contourLength][0]
            distance = math.sqrt((centerX - point[0])**2 + (centerY - point[1])**2)
            if distance <= 0:
                distance = 1
            direction = [(point[1] - centerY) / distance, (point[0] - centerX) / distance]

            intersections.append({'P': [point[1], point[0]], 'direction': np.array(direction)})
            

        warpedImg = image.copy()
        warpedSegmentation = segmentation.copy()
        factor = 1.0
        mapAggregation = None
        for intersection in intersections:
            if self.outwards:
                distanceToT = np.random.randint(r/3, r)
            else:
                distanceToT = np.random.randint(r/3, 2*r/3)
            if not self.outwards:
                factor = -1.0
            T = np.array(intersection["direction"])*factor*(distanceToT)
            intersection["T"] = (intersection["P"] + T).astype(int)
            
            warpedImg, map = localTranslationWarp(warpedImg, r, int(intersection["P"][1]), int(intersection["P"][0]), int(intersection["T"][1]), int(intersection["T"][0]), strength, True, False)
            warpedSegmentation, _ = localTranslationWarp(warpedSegmentation, r, int(intersection["P"][1]), int(intersection["P"][0]), int(intersection["T"][1]), int(intersection["T"][0]), strength, False, False)

            if mapAggregation is None:
                mapAggregation = np.zeros_like(map)
            mapAggregation = mapAggregation + map

        self.lastMap = mapAggregation

        return warpedImg, warpedSegmentation

