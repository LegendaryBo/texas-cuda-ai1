package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.io.IOException;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Interface of all classes that are used for importing individuals created by
 * some nodes in given creation cells, and exporting given individuals to some
 * place.
 * 
 * @author Marcin Golebiowski
 */
public interface EvIndividualsExchanger<T extends EvIndividual> {

  /**
   * Imports good individuals created in cells for task
   * 
   * @param cells_id cells identifiers
   * @param task_id task identifier
   * @param count how many individuals to import
   * @throws IOException
   */
  List<T> importIndividuals(long[] cells_id, long task_id, int count)
      throws IOException;


  /**
   * Export given individuals as individuals created in given cell and node for
   * task
   * 
   * @param task_id task identifier
   * @param cell_id cell identifier
   * @param node_id node identifier
   * @param individuals list of individuals to export
   * @throws IOException
   */
  void exportIndividuals(long task_id, long cell_id, long node_id,
      List<T> individuals) throws IOException;

}
