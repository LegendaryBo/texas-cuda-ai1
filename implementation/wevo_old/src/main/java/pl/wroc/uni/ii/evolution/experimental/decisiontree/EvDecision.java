package pl.wroc.uni.ii.evolution.experimental.decisiontree;

/**
 * Decision interface
 * 
 * @author Kamil Dworakowski
 */
public interface EvDecision<T> {
  EvAnswer decide(T arg);
}
