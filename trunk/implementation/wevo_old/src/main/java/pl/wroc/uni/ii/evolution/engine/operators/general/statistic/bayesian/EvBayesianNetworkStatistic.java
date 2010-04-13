package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.bayesian;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * Object that stores state of the bayesian network.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvBayesianNetworkStatistic extends EvStatistic {

  /**
   * 
   */
  private static final long serialVersionUID = -4136102765514091120L;

  /**
   * Table of size 2 x Z storing Z edges. Each 2-elemental row represents one
   * one-way edge, first integer represent index of initial initial vertex.
   */
  private int[][] edges = null;

  /**
   * Total number of vertexes.
   */
  private int number_of_vertex = 0;


  /**
   * Create object storing complete data about an bayesian network.
   * 
   * @param edg - Edges informations. Table shall be size of Z x 2, where Z is
   *        total number of edges. Each vertex is defined by number from 0 to
   *        (Z-1).<br>
   *        Each 2-elemental row represents one one-way edge, first integer
   *        represent index of initial initial vertex.
   * @param num_vertex - total number of vertexes
   */
  public EvBayesianNetworkStatistic(final int[][] edg, final int num_vertex) {
    edges = edg;
    number_of_vertex = num_vertex;
  }


  /**
   * Return edges information about bayesian network stored in the this object.
   * 
   * @return Table of size Z x 2 storing Z edges. Each 2-elemental row
   *         represents one one-way edge, first integer represent index of
   *         initial initial vertex.
   */
  public int[][] getNetwork() {
    return edges;
  }


  /**
   * @return number of vertexes in the bayesian network stored in the this
   *         object.
   */
  public int getNemberOfVertex() {
    return number_of_vertex;
  }

}
