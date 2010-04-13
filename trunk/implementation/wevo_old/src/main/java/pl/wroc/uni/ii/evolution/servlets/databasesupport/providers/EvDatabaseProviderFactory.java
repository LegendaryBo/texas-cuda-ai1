package pl.wroc.uni.ii.evolution.servlets.databasesupport.providers;

/**
 * @author Marcin Golebiowski
 */
public class EvDatabaseProviderFactory {

  /**
   * Returns instance of database provider
   */
  public static EvDatabaseProvider getProvider(String provider)
      throws InstantiationException, IllegalAccessException,
      ClassNotFoundException {
    return (EvDatabaseProvider) Class.forName(provider).newInstance();
  }
}
