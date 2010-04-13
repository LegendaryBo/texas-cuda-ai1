package pl.wroc.uni.ii.evolution.distribution2.operators;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.logging.Logger;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Simple distribution operator, selecting and sending a fixed amount of
 * individuals over the network to a given server. When applied, the operator
 * will pick as many individuals as possible (limited to the value of amount
 * parameter) and send them using HTPP protocol.
 * 
 * @author Karol Stosiek (karol.stosiek@gmail.com)
 * @param <T> EvIndividual class
 */
public class EvSimpleDistributionSendOperator<T extends EvIndividual>
    implements EvOperator<T> {

  /**
   * URL object, holding the address of the server.
   */
  private final URL url;

  /**
   * Number of individuals to select and send with each application of the
   * operator.
   */
  private final int amount;


  /**
   * Used to logging.
   */
  private final Logger logger; 
  
  /**
   * Simple operator for sending best individuals to a remote server.
   * 
   * @param initUrl - URL address of the server that collects individuals
   * @param initAmount - amount of individuals we wish to send
   */
  public EvSimpleDistributionSendOperator(final URL initUrl,
      final int initAmount) {

    this.url = initUrl;

    if (initAmount < 0) {
      throw new IllegalArgumentException(
          "It's impossible to pick less than 0 individuals.");
    }

    this.amount = initAmount;
    
    this.logger = Logger.getLogger(
        EvSimpleDistributionGetOperator.class.getCanonicalName());
  }


  /**
   * Opens a new connection with the resource targeted by the URL, for which the
   * operator was created.
   * 
   * @return connection that is created
   * @throws IOException - thrown, when unable to open connection
   */
  private HttpURLConnection openConnection() throws IOException {

    /*
     * We open a connection to a server, enable writing to server and tell proxy
     * servers not to cache the content.
     */

    this.logger.info("Setting up connection with server...");
  
    HttpURLConnection connection =
        (HttpURLConnection) this.url.openConnection();

    connection.setDoOutput(true);
    connection.setUseCaches(false);
    connection.setRequestMethod("POST");
    connection.setRequestProperty("Content-Type", "application/octet-stream");

    connection.connect(); // ensuring we are connected

    this.logger.info("Connected.");
    
    return connection;
  }


  /**
   * Closes the given connection.
   * 
   * @param connection - connection to close
   * @throws IOException - thrown, when unable to get the response
   */
  private void closeConnection(final HttpURLConnection connection)
      throws IOException {

    /*
     * We need to call these two lines so that servlets doPost method is called.
     * We ignore the response.
     */

    this.logger.info("Disconnecting...");
    
    connection.getResponseMessage();
    connection.disconnect();
    
    this.logger.info("Disconnected.");
  }


  /**
   * Writes the collection to the response body.
   * 
   * @param connection - connection with a server
   * @param selected - selected individuals to write
   * @throws IOException - thrown, when unable to write the selected individuals
   *         to the stream.
   */
  private void writeContents(final HttpURLConnection connection,
      final ArrayList<T> selected) throws IOException {

    this.logger.info("Reading response contents...");
    
    /*
     * We create an output stream to write the serialized ArrayList of
     * individuals and write them.
     */
    
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeObject(selected);
    output.flush();
    output.close();
    
    this.logger.info("Done.");
  }


  /**
   * Overridden method. See pl.wroc.uni.ii.evolution.engine for details.
   * 
   * @param population - population to apply the operator to
   * @return initial population unchanged
   */
  public EvPopulation<T> apply(final EvPopulation<T> population) {

    ArrayList<T> selected = population.kBest(this.amount);

    try {

      HttpURLConnection connection = openConnection();

      writeContents(connection, selected);

      closeConnection(connection);

      /* TODO: Introduce error handling. */ 
      
    } catch (IOException e) {
      e.printStackTrace();
    }

    return population;
  }


  /**
   * URL accessor.
   * 
   * @return the URL of the server
   */
  public URL getURL() {
    return url;
  }


  /**
   * Amount field accessor.
   * 
   * @return number of individuals to send.
   */
  public int getAmount() {
    return amount;
  }

}
