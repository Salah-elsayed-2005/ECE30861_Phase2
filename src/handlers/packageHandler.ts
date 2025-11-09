import { APIGatewayProxyEvent, APIGatewayProxyResult } from 'aws-lambda';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { 
  DynamoDBDocumentClient, 
  PutCommand, 
  GetCommand, 
  QueryCommand, 
  ScanCommand, 
  DeleteCommand,
  PutCommandInput,
  GetCommandInput,
  QueryCommandInput,
  ScanCommandInput,
  DeleteCommandInput
} from '@aws-sdk/lib-dynamodb';

/**
 * GET /tracks
 * Returns the planned tracks
 */
export const getTracks = async (event: APIGatewayProxyEvent): Promise<APIGatewayProxyResult> => {
  try {
    const tracks = {
      plannedTracks: [
        "Access control track"
      ]
    };

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify(tracks)
    };
  } catch (error) {
    console.error('Error getting tracks:', error);
    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({ error: 'Failed to get tracks' })
    };
  }
};

/**
 * DELETE /reset
 * Reset the registry by deleting all packages
 */
export const resetRegistry = async (event: APIGatewayProxyEvent): Promise<APIGatewayProxyResult> => {
  try {
    const tableName = process.env.DYNAMODB_TABLE_NAME;
    if (!tableName) {
      return {
        statusCode: 500,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify({ error: 'Database configuration error' })
      };
    }

    // Scan all items
    const scanParams: ScanCommandInput = {
      TableName: tableName,
    };

    const scanResult = await docClient.send(new ScanCommand(scanParams));

    // Delete all items
    if (scanResult.Items && scanResult.Items.length > 0) {
      for (const item of scanResult.Items) {
        const deleteParams: DeleteCommandInput = {





























};  }    };      body: JSON.stringify({ error: 'Failed to reset registry' })      },        'Access-Control-Allow-Origin': '*'        'Content-Type': 'application/json',      headers: {      statusCode: 500,    return {    console.error('Error resetting registry:', error);  } catch (error) {    };      body: JSON.stringify({ message: 'Registry reset successfully' })      },        'Access-Control-Allow-Origin': '*'        'Content-Type': 'application/json',      headers: {      statusCode: 200,    return {    }      }        await docClient.send(new DeleteCommand(deleteParams));        };          },            id: item.id,          Key: {          TableName: tableName,        'Content-Type': 'text/plain',
        'Access-Control-Allow-Origin': '*'
      },
      body: 'Registry is reset.'
    };
  } catch (error) {
    console.error('Error resetting registry:', error);
    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({ error: 'Failed to reset registry' })
    };
  }
};