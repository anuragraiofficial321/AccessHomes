Step 1: Frame Extraction

* Process the video and extract 1 frame per second.
* For example, a 1-minute video produces 60 frames.

Step 2: Object Detection in Frames

* Each frame (image) is passed through an object detection model (e.g., YOLO).
* The model identifies existing furniture and fixtures like couch, chair, toilet, sink, etc.

  * Example:

    * Frame 1 → Couch detected with 0.85 confidence.
    * Frame 2 → Chair detected with 0.92 confidence.
    * Frame 3 → Toilet detected with 0.93 confidence.

Step 3: Multimodal Model Input

* Each detected frame, along with:

  * The user prompt,
  * The user persona, and
  * The user’s requirements,
    is passed to a multimodal model.

Step 4: Object Recommendation & Placement

* The model suggests the best additional object (PNG/icon) that matches the user’s needs.
* It also provides the best placement location in the frame:

  * Either in free space (e.g., empty floor area),
  * Or near an existing object (e.g., placing a small table near the couch).

Step 5: Output

* The system returns:

  * A PNG/icon of the recommended object,
  * Details of where to place it in the floor plan or house image.

### Steps:

- Generate the yolo summary for given video 
- pass the floor plan image of cubicasa and generated floorplan and give me recommendation where to place the object on floor plan
- Pass more than one frame at a time.


-----------------------------------------------------------------------
- In the given code i am able to plot the object png and fixed furniture polygon but i also want to plot the png on fixed furniture so i can show both the png and detected and plotting object on floor plan etc

rewrite plotting code so i check the class using fixed furniture and plot the symbol on floor plan etc

-----------------------------------------------------------------------