package com.qcom.aistack_objdetect;

import java.util.ArrayList;
/**
 * RectangleBox class defines the property associated with each box like coordinates
 * labels, confidence etc.
 * Can also create copy of boxes.
 */
public class RectangleBox {

    public float top;
    public float bottom;
    public float left;
    public float right;

    public int fps;
    public String processing_time;
    public String label;
    public static ArrayList<RectangleBox> createBoxes(int num) {
        final ArrayList<RectangleBox> boxes;
        boxes = new ArrayList<>();
        for (int i = 0; i < num; ++i) {
            boxes.add(new RectangleBox());
        }
        return boxes;
    }
}
