package com.qcom.aistack_objdetect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;


import java.util.ArrayList;
import java.util.concurrent.locks.ReentrantLock;

/**
 * FragmentRender class is utility for making boxes on camera frames.
 * FragmentRender has utility in fragment_camera.xml and CameraFragment Class
 */
public class FragmentRender extends View {

    private ReentrantLock mLock = new ReentrantLock();
    private ArrayList<RectangleBox> boxlist = new ArrayList<>();

    private Paint mTextColor= new Paint();
    private Paint mBorderColor= new Paint();

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }


    public void setCoordsList(ArrayList<RectangleBox> t_boxlist) {
        mLock.lock();
        postInvalidate();

        if (boxlist==null)
        {
            mLock.unlock();
            return;
        }
        boxlist.clear();
        for(int j=0;j<t_boxlist.size();j++) {
            System.out.println("writing boxList in java");
            boxlist.add(t_boxlist.get(j));
        }
        mLock.unlock();
        postInvalidate();
    }


    private void init() {
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);

        mBorderColor.setColor(Color.TRANSPARENT);
        mBorderColor.setColor(Color.MAGENTA);
        mBorderColor.setStyle(Paint.Style.STROKE);
        mBorderColor.setStrokeWidth(6);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);
        mTextColor.setColor(Color.RED);
    }

    @Override
    protected void onDraw(Canvas canvas) {

        mLock.lock();
        System.out.println("BOX LIST SIZE:    "+boxlist.size());
        for(int j=0;j<boxlist.size();j++) {

            RectangleBox rbox = boxlist.get(j);
            float y = rbox.left;
            float y1 = rbox.right;
            float x =  rbox.top;
            float x1 = rbox.bottom;


            String fps_textLabel = "FPS: "+String.valueOf(rbox.fps);
            canvas.drawText(fps_textLabel,10,70,mTextColor);

            String processingTimeTextLabel= rbox.processing_time+"ms";

            canvas.drawRect(x1, y, x, y1, mBorderColor);
            canvas.drawText(rbox.label,x1+10, y+40, mTextColor);
            canvas.drawText(processingTimeTextLabel,x1+10, y+90, mTextColor);

        }
        mLock.unlock();
    }
}
