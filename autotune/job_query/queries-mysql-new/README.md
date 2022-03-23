select current_timestamp(6) into @query_start;
set @query_name='README';

These query variants are from https://github.com/winkyao/join-order-benchmark/tree/master/queries
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;