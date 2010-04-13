package pl.wroc.uni.ii.evolution.distribution.workers;

import java.util.Calendar;
import java.util.TimeZone;

public class EvConsole {

  private static Calendar cal = Calendar.getInstance(TimeZone.getDefault());

  private static java.text.SimpleDateFormat sdf =
      new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss");


  public static void println(Object o) {
    cal.setTimeInMillis(System.currentTimeMillis());

    String DATE_FORMAT = "yyyy-MM-dd HH:mm:ss";
    sdf = new java.text.SimpleDateFormat(DATE_FORMAT);
    sdf.setTimeZone(TimeZone.getDefault());

    String time = sdf.format(cal.getTime()) + ": ";
    System.out.println(time + ":" + o.toString());
    System.err.println(o);
  }
}
