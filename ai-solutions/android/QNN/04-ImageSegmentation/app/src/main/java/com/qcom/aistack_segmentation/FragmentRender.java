package com.qcom.aistack_segmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.os.Trace;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import java.util.concurrent.locks.ReentrantLock;


/**
 * FragmentRender class is utility for segmentation on camera frames.
 * FragmentRender has utility in fragment_camera.xml and CameraFragment Class
 */

public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    Bitmap bms = null;



    String[] LABEL_NAMES = {
            "background",  //White
            "aeroplane",   //red
            "bicycle",   //red
            "bird",   //green
            "boat",  //red
            "bottle",  //blue
            "bus",   //red
            "car",   //red
            "cat",   //green
            "chair",  //blue
            "cow",   //green
            "diningtable", //blue
            "dog",  //green
            "horse",  //green
            "motorbike",  //red
            "person",  //green
            "pottedplant", //green
            "sheep", //green
            "sofa", //blue
            "train", //red
            "tv" //blue
    };

    int[] ColorArray = {
            Color.GREEN,
            Color.RED, //aeroplane
            Color.RED, //bicycle
            Color.GREEN, //bird
            Color.RED, //boat
            Color.MAGENTA,  // 5 - bottle
            Color.RED, //bus
            Color.RED,  // 7 - car
            Color.GREEN, // cat
            Color.CYAN, // 9 - chair
            Color.GREEN, //cow
            Color.GREEN, //diningtable
            Color.GREEN, // dog
            Color.GREEN, //horse
            Color.RED, //motorbike
            Color.GREEN, // 15 - Person
            Color.WHITE, //pottedplant
            Color.GREEN, //sheep
            Color.BLUE, // sofa
            Color.RED, // 19 - train
            Color.BLUE // 20 - tv
    };


    int render_count=-1;

    int fps;
    float inferTime = 500.0f;
    float[] segmap = null;
    int width;
    int height;

    float renderTime = 500.0f;



    int[] color_points;


    private final Paint mTextColor= new Paint();

    private final Paint[] paintArray;
    private final Paint mBorderColor= new Paint();

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);


        init();
        paintArray = new Paint[21];
        for (int i = 0; i < 21; i++)
        {
            paintArray[i] = new Paint();
            paintArray[i].setColor(ColorArray[i]);
            paintArray[i].setAlpha(5);
            paintArray[i].setStrokeWidth(6);

        }


    }

    public void findpixelsfromfloataaray(float[] map, int width, int height)
    {
        Trace.beginSection("findpixelsfromfloataaray");
        mLock.lock();


        for (int i=0;i<map.length;i++)
        {
            color_points[i]=(int)(map[i]);
        }

        mLock.unlock();
        Trace.endSection();
    }


    public void set_map(float[] map,int width, int height,int fps, float inferTime)
    {
        Log.e(" ","set_map width="+width+" height="+height+" fps="+fps+" inferTime="+inferTime );

        long rStart=System.nanoTime();

        Trace.beginSection("setmap");
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.inferTime = inferTime;


        findpixelsfromfloataaray(map,width,height);

        postInvalidate();

        Trace.endSection();

        renderTime=(System.nanoTime()-rStart)/1000000.0f;
        Log.e(" ","set_map Render:"+renderTime+"ms");
    }


    private void init() {
        segmap = null;
        mBorderColor.setColor(Color.MAGENTA);
        mBorderColor.setStrokeWidth(6);
        mBorderColor.setAlpha(5);

        mTextColor.setColor(Color.RED);
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);


        color_points = new int[1296000];


    }

    @Override
    protected void onDraw(Canvas canvas) {

        Trace.beginSection("OnDraw");
        mLock.lock();

        long rStart=System.nanoTime();


        Log.e(" ","createBitmap width="+width+" height="+height+" render_count="+render_count);


        if (canvas.isHardwareAccelerated())
        {
            Log.e(" ","canvas is HardwareAccelerated");
        }
        else
        {
            Log.e(" ","canvas is NOT HardwareAccelerated");
        }


        canvas.drawBitmap(color_points,0,width,0,0,width,height,false,null);

        renderTime=(System.nanoTime()-rStart)/1000000.0f;
        canvas.drawText("Infer: "+String.format("%.2f", inferTime) + "ms", 10, 70, mTextColor);
        canvas.drawText("FPS: " + fps, 10, 120, mTextColor);
        canvas.drawText("Render: "+ String.format("%.2f", renderTime) + "ms", 10, 170, mTextColor);
        Log.e(" ","canvas Render: "+String.format("%.2f", renderTime)+"ms");

        mLock.unlock();
        Trace.endSection();
    }
}
