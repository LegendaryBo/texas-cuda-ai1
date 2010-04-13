package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency;

import java.io.FileInputStream;
import java.io.FileOutputStream;
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
public class EvPersistentStatisticFileStorage implements
    EvPersistentStatisticStorage {

  private String path;


  public EvPersistentStatisticFileStorage(String path) {
    this.path = path;
  }


  public EvStatistic[] getStatistics() {
    List<EvStatistic> result = null;

    try {
      result = loadStatistics();
    } catch (Exception ex) {

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
  public List<EvStatistic> loadStatistics() throws Exception {

    FileInputStream in_stream = new FileInputStream(path);
    ObjectInputStream in_obj = new ObjectInputStream(in_stream);

    List<EvStatistic> res = (List<EvStatistic>) in_obj.readObject();

    in_obj.close();
    in_stream.close();

    return res;
  }


  public void saveStatistics(List<EvStatistic> stats) {

    try {
      FileOutputStream out_stream = new FileOutputStream(path);
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


  public void saveStatistic(EvStatistic stats) {

    List<EvStatistic> result = null;
    try {
      result = loadStatistics();
    } catch (Exception ex) {
      result = new ArrayList<EvStatistic>();
      System.out.println("empty");
    }
    result.add(stats);

    saveStatistics(result);

  }
}
