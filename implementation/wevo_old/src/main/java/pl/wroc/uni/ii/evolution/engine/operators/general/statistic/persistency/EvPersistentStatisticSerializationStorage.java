package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * An implementation of EvPersistentStatisticStorage using Java Serialization
 * and files.
 * 
 * @author Marcin Golebiowski
 */
public class EvPersistentStatisticSerializationStorage implements
    EvPersistentStatisticStorage {

  private String directory_path;

  private int counter = 0;


  public EvPersistentStatisticSerializationStorage(String directory_path) {
    this.directory_path = directory_path;
  }


  public EvStatistic[] getStatistics() {
    List<EvStatistic> result = null;

    try {
      result = loadStatistics();
    } catch (Exception ex) {
      ex.printStackTrace();
    }
    if (result != null && result.size() != 0) {
      EvStatistic[] t_result = new EvStatistic[result.size()];

      for (int i = 0; i < result.size(); i++) {
        t_result[i] = result.get(i);
      }

      return t_result;

    } else {
      return null;
    }
  }


  @SuppressWarnings("unchecked")
  public List<EvStatistic> loadStatistics() {

    File dir = new File(directory_path);

    String[] files = dir.list(new FilenameFilter() {
      public boolean accept(File dir, String name) {
        return name.startsWith("wevo_stat");
      }
    });

    if (files == null) {
      return null;
    }

    List<EvStatistic> result = new ArrayList<EvStatistic>();

    try {
      for (String name : files) {
        FileInputStream in_stream =
            new FileInputStream(directory_path + "/" + name);
        ObjectInputStream in_obj = new ObjectInputStream(in_stream);

        EvStatistic statistic = (EvStatistic) in_obj.readObject();
        result.add(statistic);

        in_obj.close();
        in_stream.close();

      }

    } catch (Exception ex) {
      ex.printStackTrace();
    }

    return result;
  }


  public void saveStatistic(EvStatistic stats) {

    try {
      FileOutputStream out_stream =
          new FileOutputStream(directory_path + "/wevo_stat." + counter++);
      ObjectOutputStream out_obj = new ObjectOutputStream(out_stream);
      out_obj.writeObject(stats);
      out_obj.flush();
      out_obj.close();
      out_stream.flush();
      out_stream.close();
    } catch (Exception ex) {
      ex.printStackTrace();
    }

  }
}
