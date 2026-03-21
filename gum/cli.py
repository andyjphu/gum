from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import os
import argparse
import asyncio
import shutil
import sys
from gum import gum

from gum.observers.retro import Retro

from gum.config import (
    CACHE_DIR,
    RETRO_IMAGES_DIR,
    DEFAULT_NUM_PASSES,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    DEFAULT_MAX_STATES_IN_CONTEXT,
    DEFAULT_PASS_WINDOW_SIZE,
)

class QueryAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, '')
        else:
            setattr(namespace, self.dest, values)


def parse_args():
    parser = argparse.ArgumentParser(description='GUM - A Python package with command-line interface')
    parser.add_argument('--user-name', '-u', type=str, help='The user name to use')

    parser.add_argument(
        '--query', '-q',
        nargs='?',
        action=QueryAction,
        help='Query the GUM with an optional query string',
    )
    parser.add_argument(
        '--recent', '-r',
        action='store_true',
        help='List the most recent propositions instead of running BM25 search',
    )
    
    parser.add_argument('--limit', '-l', type=int, help='Limit the number of results', default=10)
    parser.add_argument('--model', '-m', type=str, help='Model to use')
    parser.add_argument('--reset-cache', action='store_true', help='Reset the GUM cache and exit')  # Add this line

    # Batching configuration arguments
    parser.add_argument('--min-batch-size', type=int, help='Minimum number of observations to trigger batch processing')
    parser.add_argument('--max-batch-size', type=int, help='Maximum number of observations per batch')

    # Multi-pass configuration arguments
    parser.add_argument(
        '--num-passes', '-p',
        type=int,
        default=None,
        help=f'Number of analysis passes (default: {DEFAULT_NUM_PASSES})'
    )
    parser.add_argument(
        '--context-window-size', '-c',
        type=int,
        default=None,
        help=f'Temporal frame window size for context (default: {DEFAULT_CONTEXT_WINDOW_SIZE})'
    )
    parser.add_argument(
        '--max-states-in-context',
        type=int,
        default=None,
        help=f'Max unique state labels per pass in multi-pass context (default: {DEFAULT_MAX_STATES_IN_CONTEXT})'
    )
    parser.add_argument(
        '--pass-window-size',
        type=int,
        default=None,
        help=f'Number of prior passes to include in context window (default: {DEFAULT_PASS_WINDOW_SIZE})'
    )

    args = parser.parse_args()

    if not hasattr(args, 'query'):
        args.query = None

    return args


async def main():
    args = parse_args()

    # Handle --reset-cache before anything else
    if getattr(args, 'reset_cache', False):
        cache_dir = CACHE_DIR # was os.path.expanduser('./.cache/gum/')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Deleted cache directory: {cache_dir}")
        else:
            print(f"Cache directory does not exist: {cache_dir}")
        return

    model = args.model or os.getenv('MODEL_NAME') or 'gpt-4o-mini'
    user_name = args.user_name or os.getenv('USER_NAME')

    # Batching configuration - follow same pattern as other args #TODO: adjust for available models
    min_batch_size = args.min_batch_size or int(os.getenv('MIN_BATCH_SIZE', '5')) # 5 -> 1 , lowered to improve API understanding
    max_batch_size = args.max_batch_size or int(os.getenv('MAX_BATCH_SIZE', '15')) # 15 -> 1

    # you need one of: user_name for listening mode, --query, or --recent
    if user_name is None and args.query is None and not getattr(args, 'recent', False):
        print("Please provide a user name (-u), a query (-q), or use --recent to list latest propositions")
        return
    
    if getattr(args, 'recent', False):
        gum_instance = gum(user_name or os.getenv('USER_NAME') or 'default', model)
        await gum_instance.connect_db()
        props = await gum_instance.recent(limit=args.limit)
        print(f"\nRecent {len(props)} propositions:")
        for p in props:
            print(f"\nProposition: {p.text}")
            if p.reasoning:
                print(f"Reasoning: {p.reasoning}")
            if p.confidence is not None:
                print(f"Confidence: {p.confidence:.2f}")
            print(f"Created At: {p.created_at}")
            print("-" * 80)
    elif args.query is not None:
        gum_instance = gum(user_name, model)
        await gum_instance.connect_db()
        result = await gum_instance.query(args.query, limit=args.limit)

        # confidences / propositions / number of items returned
        print(f"\nFound {len(result)} results:")
        for prop, score in result:
            print(f"\nProposition: {prop.text}")
            if prop.reasoning:
                print(f"Reasoning: {prop.reasoning}")
            if prop.confidence is not None:
                print(f"Confidence: {prop.confidence:.2f}")
            print(f"Relevance Score: {score:.2f}")
            print("-" * 80)
    else:
        # Multi-pass configuration
        num_passes = args.num_passes or int(os.getenv('GUM_NUM_PASSES', str(DEFAULT_NUM_PASSES)))
        context_window_size = args.context_window_size or int(os.getenv('GUM_CONTEXT_WINDOW_SIZE', str(DEFAULT_CONTEXT_WINDOW_SIZE)))
        max_states_in_context = args.max_states_in_context or int(os.getenv('GUM_MAX_STATES_IN_CONTEXT', str(DEFAULT_MAX_STATES_IN_CONTEXT)))
        pass_window_size = args.pass_window_size or int(os.getenv('GUM_PASS_WINDOW_SIZE', str(DEFAULT_PASS_WINDOW_SIZE)))

        print(f"Listening to {user_name} with model {model}")
        print(f"Multi-pass: {num_passes} passes, context window: {context_window_size}, max states in context: {max_states_in_context}, pass window: {pass_window_size}")

        retro_observer = Retro(
            model_name=model,
            debug=True,
            images_dir=RETRO_IMAGES_DIR,
            num_passes=num_passes,
            context_window_size=context_window_size,
            max_states_in_context=max_states_in_context,
            pass_window_size=pass_window_size,
        )

        async with gum(
                user_name,
                model,
                retro_observer,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size
        ) as gum_instance:
            await retro_observer.stopped.wait()

            # Check if the observer's background task failed with an exception.
            # The task runs via asyncio.create_task so exceptions are stored on
            # the Task object rather than propagating automatically.
            if retro_observer._task and retro_observer._task.done():
                exc = retro_observer._task.exception()
                if exc is not None:
                    raise exc


def cli():
    try:
        asyncio.run(main())
    except Exception as exc:
        import traceback
        msg = f"\n[FATAL] {exc}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        # Persist to mounted volume so the error survives container removal
        log_dir = os.environ.get("GUM_TRAFFIC_LOG_DIR", "")
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "FATAL.log"), "a") as f:
                    from datetime import datetime, timezone
                    f.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
            except OSError:
                pass
        sys.exit(1)


if __name__ == '__main__':
    cli()
